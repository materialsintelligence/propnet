from matminer.data_retrieval.retrieve_AFLOW import RetrievalQuery as _RetrievalQuery
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure

# noinspection PyUnresolvedReferences
import propnet.ext.aflow_redefs
from aflow.control import server as _aflow_server
from aflow import K, msg as _msg
from aflow.entries import AflowFile, Entry

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from urllib.error import HTTPError
from requests.exceptions import ConnectionError, RetryError

from concurrent.futures import ThreadPoolExecutor, as_completed
from maggma.utils import grouper
from maggma.stores import MongoStore

from propnet.core.materials import Material
from propnet.core.quantity import QuantityFactory
from propnet.core.provenance import ProvenanceElement
from propnet import ureg

import numpy as np
from datetime import datetime
import logging
from collections import defaultdict
from itertools import chain
from functools import partial

logger = logging.getLogger(__name__)


class AflowAdapter:
    """
    Retrieves AFLOW data from the AFLUX/AFLOW RESTful API or from a local MongoDB store
    to be output as propnet `Material` objects.

    References:
        AFLOW Database - Curtarolo, S. et al. http://dx.doi.org/10.1016/j.commatsci.2012.02.005
        AFLOW API - Curtarolo, S. et al. https://doi.org/10.1016/j.commatsci.2014.05.014
        AFLUX Search API - Curtarolo, S. et al. http://dx.doi.org/10.1016/j.commatsci.2017.04.036
        AFLUX API Python Wrapper - Rosenbrock, C. http://adsabs.harvard.edu/abs/2017arXiv171000813R
    """
    # Mapping is keyed by AFLOW keyword, valued by propnet property
    mapping = {
        "auid": "external_identifier_aflow",
        "Egap": "band_gap",
        "ael_bulk_modulus_reuss": "bulk_modulus",
        "ael_bulk_modulus_voigt": "bulk_modulus",
        "ael_elastic_anisotropy": "elastic_anisotropy",
        "ael_poisson_ratio": "poisson_ratio",
        "ael_shear_modulus_reuss": "shear_modulus",
        "ael_shear_modulus_voigt": "shear_modulus",
        "agl_acoustic_debye": "debye_temperature",
        "agl_debye": "debye_temperature",
        "agl_gruneisen": "gruneisen_parameter",
        "agl_heat_capacity_Cp_300K": "unit_cell_heat_capacity_constant_pressure",
        "agl_heat_capacity_Cv_300K": "unit_cell_heat_capacity_constant_volume",
        "agl_thermal_conductivity_300K": "thermal_conductivity",
        "agl_thermal_expansion_300K": "thermal_expansion_coefficient",
        "compound": "formula",
        "energy_atom": "energy_per_atom",
        "enthalpy_formation_atom": "formation_energy_per_atom",
        "structure": "structure",
        "elastic_tensor_voigt": "elastic_tensor_voigt",
        "compliance_tensor_voigt": "compliance_tensor_voigt"
    }
    """Mapping of AFLOW keywords to propnet properties. These AFLOW keywords include
    extra keywords (e.g. structure, elastic_tensor_voigt, compliance_tensor_voigt) which
    are derived from extra files attached to AFLOW entries.
    """

    property_web_file_mapping = {
        "structure": "CONTCAR.relax.vasp",
        "elastic_tensor_voigt": "AEL_elastic_tensor.json",
        "compliance_tensor_voigt": "AEL_elastic_tensor.json"
    }
    """Source files for AFLOW keywords derived from extra files attached to AFLOW entries.
    """
    web_file_property_mapping = defaultdict(list)
    """Contains a list of keywords derived from the extra file named in the key.
    """
    for k, v in property_web_file_mapping.items():
        web_file_property_mapping[v].append(k)

    property_store_field_mapping = {
        "structure": ["CONTCAR_relax_vasp", "geometry", "species",
                      "composition", "positions_fractional"],
        "elastic_tensor_voigt": ["AEL_elastic_tensor_json"],
        "compliance_tensor_voigt": ["AEL_elastic_tensor_json"]
    }
    """Contains a list of extra fields to return from a MongoDB store when the keyed
    AFLOW keyword is requested.
    """

    transform_func = {
        "energy_atom": lambda x: abs(x),
        "compound": lambda x: Composition(x).reduced_formula
    }
    """Contains references to simple functions to transform data from
    the native AFLOW format to the standard used by propnet (e.g. convert
    negative total energy to positive). Function must take the raw AFLOW value
    (as returned from Entry.keyword) as a single argument and return the transformed value.
    """

    file_transform_func = {
        "structure": lambda x: AflowAdapter._get_structure(x),
        "compliance_tensor_voigt": lambda x: AflowAdapter._get_elastic_data(x, prop='compliance'),
        "elastic_tensor_voigt": lambda x: AflowAdapter._get_elastic_data(x, prop='stiffness')
    }
    """Contains references to functions to transform data acquired from AFLOW files to their respective
    properties. Functions must take an Entry object, initialized with all the raw data necessary. File data
    is keyed by the file name, substituting '.' for '_' (e.g. 'CONTCAR.relax.vasp' -> 'CONTCAR_relax_vasp').
    Functions return the data to be stored in that property.
    
    Note: if a function exists in `transform_func` for the property, the value returned from the function 
    in `file_transform_func` may later be passed through that function as well.
    """

    unit_map = {
        "Egap": "eV",
        "ael_bulk_modulus_reuss": "gigapascal",
        "ael_bulk_modulus_voigt": "gigapascal",
        "ael_elastic_anisotropy": "dimensionless",
        "ael_poisson_ratio": "dimensionless",
        "ael_shear_modulus_reuss": "gigapascal",
        "ael_shear_modulus_voigt": "gigapascal",
        "agl_acoustic_debye": "kelvin",
        "agl_debye": "kelvin",
        "agl_gruneisen": "dimensionless",
        "agl_heat_capacity_Cp_300K": "boltzmann_constant",
        "agl_heat_capacity_Cv_300K": "boltzmann_constant",
        "agl_thermal_conductivity_300K": "watt/meter/kelvin",
        "agl_thermal_expansion_300K": "1/kelvin",
        "energy_atom": "eV/atom",
        "enthalpy_formation_atom": "eV/atom",
        "elastic_tensor_voigt": "gigapascal",
        "compliance_tensor_voigt": "gigapascal"
    }
    """Contains the units of the AFLOW data as a `pint` unit string, keyed by AFLOW keyword.
    """

    def __init__(self, store=None):
        """
        Initializes the AFLOW adapter.

        Args:
            store (maggma.stores.Store): optional, a maggma Mongo-like store containing a local
                copy of the AFLOW database. If specified, this store will be used as the source
                of AFLOW data, and no requests to the AFLUX/AFLOW API will be made.
        """
        if store is None:
            self._executor = ThreadPoolExecutor(max_workers=10)
            self.store = None
        else:
            self._executor = None
            self.store = store
            store.connect()
            self.file_transform_func['structure'] = partial(AflowAdapter._get_structure,
                                                            use_web_api=False)
        super(AflowAdapter, self).__init__()

    def __del__(self):
        if self._executor:
            self._executor.shutdown()
    
    def get_material_by_auid(self, auid):
        """
        Return a propnet material for AFLOW entry specified by the provided auid.

        Args:
            auid (str): AFLOW auid to look up

        Returns:
            propnet.core.materials.Material: propnet material of the AFLOW entry
        """
        return self.get_materials_by_auids([auid])[0]
    
    def get_materials_by_auids(self, auids, max_request_size=1000):
        """
        Returns a propnet material for each AFLOW entry specified by the provided auids.

        Args:
            auids (`list` of `str`): list of AFLOW auids
            max_request_size (int): maxmimum number of materials to retrieve per request
                (only applies to AFLUX queries). If total number of records is greater than
                 `max_request_size`, multiple requests will be made. Default: 1000

        Returns:
            `list` of `propnet.core.materials.Material`: propnet materials for each AFLOW entry,
                maintaining order of auids
        """

        materials = [m for m in self.generate_materials_by_auids(list(auids), max_request_size)]
        materials_by_auid = dict()
        for material in materials:
            auid = material.symbol_quantities_dict['external_identifier_aflow']
            materials_by_auid[auid] = material

        return [materials_by_auid[auid] for auid in auids]

    def generate_materials_by_auids(self, auids, max_request_size=1000):
        """
        Produces a generator of materials from a list of auids. Not guaranteed to be in order.
        This can be faster than get_materials_by_auids() when API requests take a long time
        as it will produce materials as they finish downloading/processing.

        Args:
            auids (`list` of `str`): list of AFLOW auids
            max_request_size (int): maxmimum number of materials to retrieve per request
                (only applies to AFLUX queries). If total number of records is greater than 
                `max_request_size`, multiple requests will be made. Default: 1000

        Returns:
            generator: produces propnet materials for each AFLOW entry
        """
        criteria = {'auid': {'$in': list(auids)}}
        properties = list(self.mapping.keys())
        if self.store is not None:
            yield from self.get_materials_from_store(criteria, properties)
        else:
            yield from self.get_materials_from_web(criteria, properties, max_request_size)
    
    @staticmethod
    def generate_all_auids(max_request_size=1000, with_metadata=False):
        """
        Produces all AUIDs stored in the AFLOW database. Note this only uses the web API.
        To produce from the store, use generate_all_auids_from_store().

        Note: large values of `max_request_size` may cause API query failures and repeated
            attempts to retrieve the data.

        Args:
            max_request_size (int): maxmimum number of entries to retrieve per request. 
                If total number of records is greater than `max_request_size`, multiple
                requests will be made. Default: 1000
            with_metadata (bool): if True, request "aurl", "compound" (formula) and "aflowlib_date"
                fields and return data as a dict.

        Returns:
            generator: produces an auid string, or with `with_metadata=True`, a dict keyed
                by AFLOW keyword
        """

        # Breaking up the query by catalog makes it faster and less prone to failure
        catalogs = ['icsd', 'lib1', 'lib2', 'lib3']
        props = ['auid']
        if with_metadata:
            props += ['aurl', 'compound', 'aflowlib_date']
        for catalog in catalogs:
            query = AflowAPIQuery.from_pymongo(
                criteria={},
                properties=props,
                request_size=max_request_size,
                catalog=catalog,
                batch_reduction=True
            )
            if not with_metadata:
                from aflow.keywords import aurl, compound, reset
                reset()
                query.exclude(aurl, compound)

            for item in query:
                if with_metadata:
                    yield item.raw
                else:
                    yield item.auid

    def generate_all_auids_from_store(self, with_metadata=False):
        """
        Produces all AUIDs stored in the local MongoDB store of the AFLOW database.
        To produce from the AFLUX API, use generate_all_auids().

        Args:
            with_metadata (bool): if True, request "aurl", "compound" (formula) and "aflowlib_date"
                fields and return data as a dict.

        Returns:
            generator: produces an auid string, or with `with_metadata=True`, a dict keyed
                by AFLOW keyword
        """
        if not self.store:
            raise ValueError("No store specified!")

        props = ['auid']
        if with_metadata:
            props += ['aurl', 'compound', 'aflowlib_date']
        query = self.store.query(
            criteria={},
            properties=props
        )

        for item in query:
            if with_metadata:
                item.pop('_id')
                yield item
            else:
                yield item['auid']

    def get_materials_from_store(self, criteria, properties, **kwargs):
        """
        Produces propnet materials from MongoDB AFLOW database using a Mongo-like query
        construction. To use the web API, use get_materials_from_web().
        
        Note: if `properties` is empty, only the AFLOW keywords listed in 
            AflowAdapter.mapping will be retrieved. If an unmappable
            keyword is specified, it will be ignored.
        
        Args:
            criteria (dict): Mongo-like query criteria
            properties (`list` or `dict`): list of fields to retrieve or
                Mongo-like projection dictionary (e.g. `{'field': True}`)
            **kwargs: arguments to MongoStore object

        Returns:
            generator: generates propnet Material objects from each record returned
                by the MongoDB query
        """
        if not self.store:
            raise ValueError("No store specified!")
        if not properties:
            properties = list(self.mapping.keys())
        for data in self.get_properties_from_store(
                criteria, properties + ['aflowlib_date'], **kwargs):
            yield self.transform_properties_to_material(data)

    def get_materials_from_web(self, criteria, properties, max_request_size=1000):
        """
        Produces propnet materials from AFLUX API using a Mongo-like query
        construction. To use a MongoDB store, use get_materials_from_store().
        
        Note: `criteria` cannot be complex. Simple equality, inequality, or '$in'
            schema are supported.
        
        Note: if `properties` is empty, only the AFLOW keywords listed in 
            AflowAdapter.mapping will be retrieved. If a valid, but unmappable
            AFLOW keyword is specified, it will be ignored.

        Args:
            criteria (dict): Mongo-like query criteria. Must be simple.
            properties (`list`): list of fields to retrieve. Does not
                support MongoDB projection dictionary.
            max_request_size (int): maxmimum number of materials to retrieve per request. 
                If total number of records is greater than `max_request_size`, multiple
                requests will be made. Default: 1000

        Returns:
            generator: generates propnet Material objects from each record returned
                by the query
        """
        if not properties:
            properties = list(self.mapping.keys())
        for data in self.get_properties_from_web(
                criteria, properties + ['aflowlib_date'], max_request_size=max_request_size):
            yield self.transform_properties_to_material(data)

    def get_properties_from_store(self, criteria, properties, **kwargs):
        """
        Produces raw property data from a MongoDB AFLOW database using a Mongo-like query
        construction. To use the web API, use get_properties_from_web().
        
        Note: if `properties` is empty, only the AFLOW keywords listed in 
            AflowAdapter.mapping will be retrieved.
            
        Args:
            criteria (dict): Mongo-like query criteria
            properties (`list` or `dict`): list of fields to retrieve or
                Mongo-like projection dictionary (e.g. `{'field': True}`)
            **kwargs: arguments to MongoStore object

        Returns:
            generator: generates dicts of data, keyed by AFLOW keyword.
        """
        if not properties:
            properties = list(self.mapping.keys())
        properties_to_retrieve = set(properties)
        file_properties_to_map = dict()
        for p, fn in self.property_store_field_mapping.items():
            if p in properties_to_retrieve:
                additional_fields = self.property_store_field_mapping[p]
                file_properties_to_map[p] = additional_fields
                properties_to_retrieve.remove(p)
                properties_to_retrieve = properties_to_retrieve.union(set(additional_fields))

        q = self.store.query(criteria=criteria, properties=properties_to_retrieve, **kwargs)
        for raw_data in q:
            raw_data.pop('_id')
            for prop in file_properties_to_map.keys():
                transformed_data = self.file_transform_func[prop](Entry(**raw_data))
                if transformed_data is not None:
                    raw_data[prop] = transformed_data
            yield self._convert_entry_to_dict(Entry(**raw_data), props=properties)

    def get_properties_from_web(self, criteria, properties, max_request_size=1000):
        """
        Produces raw property data from the AFLUX API using a Mongo-like query
        construction. To use a MongoDB store, use get_properties_from_store().

        Note: `criteria` cannot be complex. Simple equality, inequality, or '$in'
            schema are supported.

        Note: if `properties` is empty, only the AFLOW keywords listed in
            AflowAdapter.mapping will be retrieved. If a valid, but unmappable
            AFLOW keyword is specified, it will be ignored.

        Note: this method uses threads to download external files if a specified
            property depends on an external AFLOW file.

        Args:
            criteria (dict): Mongo-like query criteria. Must be simple.
            properties (`list`): list of fields to retrieve. Does not
                support MongoDB projection dictionary.
            max_request_size (int): maxmimum number of materials to retrieve per request.
                If total number of records is greater than `max_request_size`, multiple
                requests will be made. Note that external file downloads will
                be limited to 10 concurrent connections, and is not related to this
                keyword. Default: 1000

        Returns:
            generator: generates dicts of data, keyed by AFLOW keyword.
        """
        if not properties:
            properties = list(self.mapping.keys())
        files_to_download = defaultdict(list)
        properties_to_retrieve = set(properties)
        for p, fn in self.property_web_file_mapping.items():
            if p in properties:
                files_to_download[fn].append(p)
                properties_to_retrieve.remove(p)

        q = AflowAPIQuery.from_pymongo(criteria, list(properties_to_retrieve), max_request_size,
                                       batch_reduction=True, property_reduction=True)

        futures = []
        materials = dict()
        files = defaultdict(dict)

        for material in q:
            auid = material.auid
            materials[auid] = material.raw
            for filename in files_to_download:
                future = self._executor.submit(
                    self._get_aflow_file,
                    material.aurl, filename, auid=material.auid,
                    with_metadata=True
                )
                futures.append(future)

        if futures:
            for future in as_completed(futures):
                response, auid, filename = future.result()
                if isinstance(response, HTTPError):
                    logger.info("Encountered error downloading file "
                                "{} for {}:\n{}".format(filename, auid, str(response)))
                    response = None
                files[auid].update({filename: response})

                if len(files[auid]) == len(files_to_download):
                    for fn, props in files_to_download.items():
                        data_in = materials[auid].copy()
                        fn_mongo = fn.replace('.', '_')
                        data_in[fn_mongo] = files[auid][fn]
                        for prop in props:
                            transformed_data = self.file_transform_func[prop](Entry(**data_in))
                            if transformed_data is not None:
                                materials[auid][prop] = transformed_data
                    yield self._convert_entry_to_dict(Entry(**materials[auid]))
        else:
            for material in materials:
                yield self._convert_entry_to_dict(Entry(**material))

    @staticmethod
    def _convert_entry_to_dict(entry, props=None):
        """
        Converts an Entry object returned by the AFLUX Python wrapper into a dictionary.

        Args:
            entry (aflow.entry.Entry): material data as an Entry
            props (`list` of `str`): properties to return in dictionary.
                Default: None (all properties contained in Entry will be returned)

        Returns:
            dict: material data, converted into Python objects, keyed by AFLOW keyword.
                If data keyed as non-AFLOW keywords are found, they will be returned as is.
        """
        if props is None:
            props = list(entry.raw.keys())

        data = {prop: getattr(entry, prop)
                if hasattr(entry, prop)
                else entry.raw.get(prop)
                for prop in props}
        return data

    def transform_properties_to_material(self, material_data):
        """
        Produces a propnet Material object from a dictionary of AFLOW materials data.

        Args:
            material_data (dict): AFLOW materials data, keyed by AFLOW keyword

        Returns:
            propnet.core.materials.Material: propnet material containing the AFLOW data
        """
        qs = []
        auid = material_data.get('auid')
        date_created = material_data.get('aflowlib_date')
        if date_created:
            date, tz = date_created.rsplit("GMT", 1)
            tz = "GMT{:+05d}".format(int(tz) * 100)
            date_object = datetime.strptime(date + tz, "%Y%m%d_%H:%M:%S_%Z%z")
            date_created = date_object.strftime("%Y-%m-%d %H:%M:%S")

        for prop, value in material_data.items():
            if value is not None and prop in self.mapping:
                provenance = ProvenanceElement(
                    source={'source': 'AFLOW',
                            'source_key': auid,
                            'date_created': date_created}
                )

                if prop in self.transform_func:
                    value = self.transform_func[prop](value)
                if value is None:
                    continue
                q = QuantityFactory.create_quantity(
                    self.mapping.get(prop),
                    value,
                    units=self.unit_map.get(prop), provenance=provenance
                )
                qs.append(q)

        return Material(qs)

    @staticmethod
    def _get_aflow_file(aurl, filename, auid=None, with_metadata=False):
        """
        Downloads a file by name given the aurl of the file. May return an
        error if the operation was unsuccessful.

        Args:
            aurl (str): the aurl of the material
            filename (str): the name of the file to download
            auid (str): optional, the auid identifier, can be returned as metadata
            with_metadata (bool): True returns data as a tuple including the auid and filename.
                Default: False (return data only)

        Returns:
            `str` or `tuple` or `HTTPError`: If `with_metadata=False`,
                returns a string containing the file data or an `HTTPError` if unsuccessful.
                If `with_metadata=True`, returns a tuple (file data/error, auid, filename).
        """
        aff = AflowFile(aurl, filename)
        try:
            data = aff()
        except HTTPError as ex:
            data = ex
        if with_metadata:
            return data, auid, filename
        return data

    @staticmethod
    def _get_elastic_data(entry, prop=None):
        """
        Extracts compliance and stiffness tensor from AFLOW AEL tensor file. Returns
        tensors in Voigt notation.

        Args:
            entry (aflow.entries.Entry): entry containing the tensor data as a string
                or JSON dict in the "AEL_elastic_tensor_json" field
            prop (`str`): optional, specifies which tensor to return. Possible values:
                `compliance`, `stiffness`, None (default, returns both)

        Returns:
            `dict` or `pint.Quantity`: If `prop` was specified, contains a pint Quantity
                representing the tensor. If `prop=None`, contains a dict of pint Quantity
                objects, keyed by tensor name `compliance_tensor_voigt` and
                `elastic_tensor_voigt` (stiffness tensor)
        """
        data_in = entry.raw.get('AEL_elastic_tensor_json')

        if isinstance(data_in, str):
            import json
            data_in = json.loads(data_in)

        if data_in is None:
            c_tensor = None
            s_tensor = None
        else:
            units = data_in['units']

            c_tensor_in = data_in['elastic_compliance_tensor']
            s_tensor_in = data_in['elastic_stiffness_tensor']

            c_idx = [idx for idx in ['s_'+str(i)+str(j) for i in range(1, 7) for j in range(1, 7)]]
            s_idx = [idx for idx in ['c_'+str(i)+str(j) for i in range(1, 7) for j in range(1, 7)]]

            c_tensor = np.reshape([c_tensor_in[idx] for idx in c_idx], (6, 6))
            s_tensor = np.reshape([s_tensor_in[idx] for idx in s_idx], (6, 6))

            c_tensor = ureg.Quantity(c_tensor, units).to(AflowAdapter.unit_map['compliance_tensor_voigt'])
            s_tensor = ureg.Quantity(s_tensor, units).to(AflowAdapter.unit_map['elastic_tensor_voigt'])

        if prop == 'compliance':
            return c_tensor
        elif prop == 'stiffness':
            return s_tensor
        return {
            'compliance_tensor_voigt': c_tensor,
            'elastic_tensor_voigt': s_tensor
        }

    @staticmethod
    def _get_structure(entry, use_web_api=True):
        """
        Extracts structure data from AFLOW CONTCAR file. If the file is unavailable,
        the structure can be reconstituted from different fields in the AFLOW database.
        Can perform web queries to retrieve those fields.

        Args:
            entry (aflow.entries.Entry): entry containing the CONTCAR data as a string in
                the "AEL_elastic_tensor_json" field, or containing the fields 'geometry',
                'species', 'composition', and 'positions_fractional'.
            use_web_api (bool): True allows missing data to be retrieved from the AFLOW API.
                False will raise an exception if the data is missing. Default: True

        Returns:
            pymatgen.core.structure.Structure: a pymatgen structure object
        """
        data_in = entry.raw.get('CONTCAR_relax_vasp')
        contcar = True
        if data_in is None:
            contcar = False

        if contcar:
            try:
                structure = Structure.from_str(data_in, fmt="poscar")
                return structure
            except Exception:
                logger.warning("Parsing structure file for {} failed.".format(entry.auid) +
                               " Generating from material entry")

        if not use_web_api:
            if not all(kw in entry.attributes for kw in ('geometry', 'species',
                                                         'composition', 'positions_fractional')):
                return None

        from pymatgen.core.lattice import Lattice
        # This lazy-loads the data by making an HTTP request for each property it needs
        # if the fields don't exist in the entry
        geometry = entry.geometry
        lattice = Lattice.from_parameters(*geometry)

        elements = list(map(str.strip, entry.species))
        composition = list(entry.composition)
        species = list(chain.from_iterable([elem] * comp for elem, comp in zip(elements, composition)))

        xyz = entry.positions_fractional.tolist()

        return Structure(lattice, species, xyz)


class AflowAPIQuery(_RetrievalQuery):
    """
    Interface to AFLUX API Python wrapper. Extends functionality of base Query
    by allowing automatic reduction of query size upon failure and automatic retrying of API requests.
    """
    def __init__(self, *args, batch_reduction=True, property_reduction=False,
                 **kwargs):
        self._auto_adjust_batch_size = batch_reduction
        self._auto_adjust_num_props = property_reduction

        self._session = requests.Session()
        retries = Retry(total=3, backoff_factor=10, status_forcelist=[500], connect=0)
        self._session.mount('http://', HTTPAdapter(max_retries=retries))

        super(AflowAPIQuery, self).__init__(*args, **kwargs)

    @classmethod
    def from_pymongo(cls, criteria, properties, request_size, **kwargs):
        """Generates an aflow Query object from pymongo-like arguments.

        Note: This function is only re-implmeneted here until matminer's next release.

        Args:
            criteria: (dict) Pymongo-like query operator. See the
                AFLOWDataRetrieval.get_DataFrame method for more details
            properties: (list of str) Properties returned in the DataFrame.
                See the api link for a list of supported properties.
            request_size: (int) Number of results to return per HTTP request.
                Note that this is similar to "limit" in pymongo.find.
        """
        # initializes query
        query = cls(batch_size=request_size, **kwargs)

        # adds filters to query
        query._add_filters(criteria)

        # determines properties returned by query
        query.select(*[getattr(K, i) for i in properties])

        # suppresses properties that may have been included as search criteria
        # but are not requested properties, which the user wants returned
        excluded_keywords = set(criteria.keys()) - set(properties)
        query.exclude(*[getattr(K, i) for i in excluded_keywords])

        return query
        
    def _request(self, n, k):
        """Constructs the query string for this `AflowAPIQuery` object for the
        specified paging limits and then returns the response from the REST API
        as a python object. Reduces batch size and/or number of properties
        automatically if query fails, reconstructing the nth page of k results
        from smaller queries.

        Args:
            n (int): page number of the results to return.
            k (int): number of datasets per page.
        """
        if len(self.responses) == 0:
            # We are making the very first request, finalize the query.
            self.finalize()
            
        server = _aflow_server
        matchbook = self.matchbook()
        directives = self._directives(n, k)
        request_url = self._get_request_url(server, matchbook, directives)
        logger.debug('Requesting page {} with {} records from url:\n{}'.format(n, k, request_url))
        try:
            is_ok, response = self._get_response(request_url, session=self._session)
        except ConnectionError as ex:
            # We requested SO many things that the server rejected our request
            # outright as opposed to trying to complete the request and failing
            is_ok = False
            response = ex.args

        if not is_ok:
            if self._auto_adjust_batch_size and self._auto_adjust_num_props:
                response = self._request_with_fewer_props(n, k, reduce_batch_on_fail=True)
            elif self._auto_adjust_batch_size:
                response = self._request_with_smaller_batch(n, k)
            elif self._auto_adjust_num_props:
                response = self._request_with_fewer_props(n, k)
            else:
                raise ValueError("The API failed to complete the request.")

        if not response:
            self._N = 0
            raise ValueError("Empty response from URI. "
                             "Check your query filters.\nURI: {}".format(request_url))

        # If this is the first request, then save the number of results in the
        # query.
        if len(self.responses) == 0:
            self._N = int(next(iter(response.keys())).split()[-1])

        # Filter out any extra responses that we got
        collected_responses = {kk: v for kk, v in response.items()
                               if int(kk.split()[0]) <= n*k}
        self.responses[n] = collected_responses

    def _request_with_fewer_props(self, n, k, reduce_batch_on_fail=False):
        """
        Requests the nth page of k results from the AFLUX API, using batches of properties.

        The algorithm divides the number of properties into x chunks, starting with x = 2,
        and requests each chunk. If one of the chunks fails, optionally, the batch size is
        reduced according to `_request_with_smaller_batch()`. If the chunk continues to fail,
        x is increased by 1, the properties are re-chunked and re-requested. This proceeds
        until each chunk contains only one property. If the query still fails, an error is raised.

        Args:
            n (int): page number of the results to return.
            k (int): number of datasets per page.
            reduce_batch_on_fail (bool): True causes batch size to decrease if a query fails
                to produce results prior to decreasing the chunk size. False does not decrease
                the batch size. Default: False

        Returns:
            dict: cumulative response from API
        """
        collected_responses = defaultdict(dict)
        props = self.selects
        chunks = 2
        while len(props) // chunks >= 1:
            if len(props) / chunks < 2:
                chunks = len(props) + 1
            query_error = False
            for chunk in grouper(props, (len(props) // chunks) + 1):
                logger.debug('Requesting property chunk {} with {} records'.format(chunks, k))
                props_to_request = set(c for c in chunk if c is not None)
                props_to_request.add(str(self.order))
                query = AflowAPIQuery.from_pymongo(criteria={},
                                                   properties=list(props_to_request),
                                                   request_size=k,
                                                   batch_reduction=reduce_batch_on_fail)
                query.filters = self.filters
                query.orderby(self.order, self.reverse)
                query._session = self._session
                try:
                    query._request(n, k)
                except ValueError:
                    query_error = True
                if not query_error:
                    response = query.responses[n]
                    for record_key, record in response.items():
                        collected_responses[record_key].update(record)
                else:
                    break

            if query_error:
                chunks += 1
            else:
                return collected_responses

        raise ValueError("The API failed to complete the request "
                         "and reducing the number of properties failed to fix it.")

    def _request_with_smaller_batch(self, original_n, original_k):
        """
        Requests the nth page of k results from the AFLUX API by requesting a smaller number
        of datasets.

        The algorithm divides the original number of datasets (`original_k`) in half and requests
        each half separately. If one of these requests fails, the number of datasets is cut
        in half again and each half is requested. The number of datasets is reduced until
        only one dataset is requested. If that fails, an error is raised.

        Args:
            original_n (int): page number of the results to return.
            original_k (int): number of datasets per page.

        Returns:
            dict: cumulative response from API
        """
        collected_responses = {}

        n, k, n_pages = self._get_next_paging_set(original_n, original_k, original_n, original_k)

        # This logic reduces the requested batch size if we experience errors
        while n_pages > 0:
            logger.debug('Requesting page {} with {} records'.format(n, k))
            server = _aflow_server
            matchbook = self.matchbook()
            directives = self._directives(n, k)
            request_url = self._get_request_url(server, matchbook, directives)
            try:
                is_ok, response = self._get_response(request_url, session=self._session)
            except (ConnectionError, RetryError) as ex:
                # We requested SO many things that the server rejected our request
                # outright as opposed to trying to complete the request and failing
                is_ok = False
                response = ex.args

            if not is_ok:
                n, k, n_pages = self._get_next_paging_set(n, k,
                                                          original_n, original_k)
            else:
                n += 1
                n_pages -= 1
                collected_responses.update(response)

            if k == 0:
                raise ValueError("The API failed to complete the request "
                                 "and reducing the batch size failed to fix it.")

        return collected_responses

    @staticmethod
    def _get_next_paging_set(n, k, original_n, original_k):
        """
        Calculates the page number and dataset size for the batch reduction algorithm.

        Args:
            n (int): current page number
            k (int): current batch size
            original_n (int): page number of original request (before batch reduction)
            original_k (int): batch size of original request (before batch reduction)

        Returns:
            tuple: (new n, new k, new # pages) where "new n" is the new starting page
                number, "new k" is the new batch size, and "new # pages" is the number
                of pages needed to fulfill the original request.
        """
        starting_entry = (n-1)*k+1
        last_entry = original_n*original_k
        new_k = k // 2
        new_n = starting_entry // new_k + 1
        new_pages = (last_entry-starting_entry) // new_k + 1
        return new_n, new_k, new_pages
        
    @staticmethod
    def _get_response(url, session=None, page=None):
        """
        Retrieve JSON data from URL with retries.

        Args:
            url (str): url to retrieve
            session (requests.Session): optional, a Session object to
                perform the request
            page (int): optional, metadata holding the page number of the request.
                Useful when this method is called using a thread pool.

        Returns:
            tuple: if page is specified, (is_ok, response, page), otherwise (is_ok, response),
                where `is_ok` is True if the request completed successfully, False otherwise,
                `response` contains the JSON response or the error information if the request
                was unsuccessful, `page` is the page number metadata.
        """
        if not session:
            session = requests.Session()
            retries = Retry(total=5, backoff_factor=10, status_forcelist=[500], connect=0)
            session.mount('http://', HTTPAdapter(max_retries=retries))
        try:
            rawresp = session.get(url)
            is_ok = rawresp.ok
            response = rawresp.json()
        except (ConnectionError, RetryError) as ex:
            is_ok = False
            response = ex.args
            _msg.err("{}\n\n{}".format(url, response))

        if page is not None:
            return is_ok, response, page
        return is_ok, response

    @staticmethod
    def _get_request_url(server, matchbook, directives):
        """
        Constructs AFLUX API URL

        Args:
            server (str): API server address
            matchbook (str): AFLUX matchbook string
            directives (str): AFLUX directives (paging, format, etc.)

        Returns:
            str: the full query URL
        """
        return "{0}{1},{2}".format(server, matchbook, directives)
