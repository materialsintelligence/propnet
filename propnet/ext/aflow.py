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
    property_file_mapping = {
        "structure": "CONTCAR.relax.vasp",
        "elastic_tensor_voigt": "AEL_elastic_tensor.json",
        "compliance_tensor_voigt": "AEL_elastic_tensor.json"
    }
    file_property_mapping = {v: k for k, v in property_file_mapping.items()}

    transform_func = {
        "energy_atom": lambda x: abs(x),
        "compound": lambda x: Composition(x).reduced_formula,
        "structure": lambda x: AflowAdapter._transform_structure(x),
        "compliance_tensor_voigt": lambda x: AflowAdapter._transform_elastic_tensor(x, prop='compliance'),
        "elastic_tensor_voigt": lambda x: AflowAdapter._transform_elastic_tensor(x, prop='stiffness')
    }
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

    def __init__(self, max_sim_requests=10, store=None):
        if store is None:
            self._executor = ThreadPoolExecutor(max_workers=max_sim_requests)
            self.store = None
            self._cache = dict()
        else:
            self._executor = None
            self.store = store
            self.transform_func['structure'] = partial(AflowAdapter._transform_structure,
                                                       store=store)
            store.connect()
            self._cache = None
        super(AflowAdapter, self).__init__()

    def __del__(self):
        if self._executor:
            self._executor.shutdown()
    
    def get_material_by_auid(self, auid):
        return self.get_materials_by_auids([auid])[0]
    
    def get_materials_by_auids(self, auids, max_request_size=1000):
        return [m for m in self.generate_materials_by_auids(auids, max_request_size)]
    
    def generate_materials_by_auids(self, auids, max_request_size=1000):
        criteria = {'auid': {'$in': auids}}
        properties = list(self.mapping.keys())
        if self.store is not None:
            yield from self.get_materials_from_store(criteria, properties)
        else:
            yield from self.get_materials_from_web(criteria, properties, max_request_size)
    
    @staticmethod
    def generate_all_auids(max_request_size=1000, with_metadata=False):
        props = ['auid']
        if with_metadata:
            props += ['aurl', 'compound', 'aflowlib_date']
        query = AflowAPIQuery.from_pymongo(
            criteria={},
            properties=props,
            request_size=max_request_size
        )

        for item in query:
            if with_metadata:
                yield item.raw
            else:
                yield item.auid

    def get_materials_from_store(self, criteria, properties, **kwargs):
        if not self.store:
            raise ValueError("No store specified!")
        if not properties:
            properties = list(self.mapping.keys())
        properties += ['aflowlib_date']
        for data in self.get_properties_from_store(criteria, properties, **kwargs):
            yield self.transform_properties_to_material(data)

    def get_materials_from_web(self, criteria, properties, max_request_size=1000):
        if not properties:
            properties = list(self.mapping.keys())
        properties += ['aflowlib_date']
        for data in self.get_properties_from_web(criteria, properties,
                                                 max_request_size=max_request_size):
            yield self.transform_properties_to_material(data)

    def get_properties_from_store(self, criteria, properties, **kwargs):
        file_properties_to_map = defaultdict(list)
        for p, fn in self.property_file_mapping.items():
            if p in properties:
                mongo_field_name = fn.replace('.', '_')
                file_properties_to_map[mongo_field_name].append(p)
                properties.remove(p)
                properties.append(mongo_field_name)

        q = self.store.query(criteria=criteria, properties=properties, **kwargs)
        for raw_data in q:
            raw_data.pop('_id')
            data = dict()
            for field, props in file_properties_to_map.items():
                field_data = raw_data.get(field)
                if field_data:
                    data.update({
                        prop: field_data for prop in props
                    })
                    raw_data.pop(field)
            entry = Entry(**raw_data)
            for prop in entry.attributes:
                data[prop] = getattr(entry, prop)
            yield data

    def get_properties_from_web(self, criteria, properties, max_request_size=1000):
        files_to_download = defaultdict(list)
        for p, fn in self.property_file_mapping.items():
            if p in properties:
                files_to_download[fn].append(p)
                properties.remove(p)

        q = AflowAPIQuery.from_pymongo(criteria, properties, max_request_size,
                                       batch_reduction=True, property_reduction=True)
        if not files_to_download:
            for material in q:
                data = dict()
                for prop in material.attributes:
                    data[prop] = getattr(material, prop)
                yield data
            raise StopIteration

        futures = []
        materials = dict()
        files = defaultdict(dict)

        for material in q:
            auid = material.auid
            data = dict()
            for prop in material.attributes:
                data[prop] = getattr(material, prop)
            materials[auid] = data
            for filename in files_to_download:
                future = self._executor.submit(
                    self._get_aflow_file,
                    material.auid, material.aurl, filename
                )
                futures.append(future)

        for future in as_completed(futures):
            auid, filename, response = future.result()
            if isinstance(response, HTTPError):
                logger.info("Encountered error downloading file "
                            "{} for {}:\n{}".format(filename, auid, str(response)))
                response = auid
            files[auid].update({filename: response})

            if len(files[auid]) == len(files_to_download):
                materials[auid].update({prop: file_data
                                        for fn, file_data in files[auid].items()
                                        for prop in files_to_download[fn]
                                        if file_data is not None})
                yield materials[auid]

    def _get_aflow_file(self, auid, aurl, filename):
        aff = AflowFile(aurl, filename)
        try:
            data = aff()
        except HTTPError as ex:
            return auid, filename, ex
        return auid, filename, data

    def transform_properties_to_material(self, material_data):
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
    def _transform_elastic_tensor(data_in, prop=None):
        if isinstance(data_in, str):
            if data_in.startswith("aflow"):
                # We got an auid because there's no tensor file
                return None
            import json
            data_in = json.loads(data_in)
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
    def _transform_structure(data_in, store=None):
        # Try to convert data to structure assuming data_in contains
        # a string of CONTCAR data.
        try:
            structure = Structure.from_str(data_in, fmt="poscar")
            return structure
        except Exception:
            logger.warning("No structure file for {}. Generating from material entry".format(data_in))

        # If that fails, either because the file was not formatted correctly,
        # or there was no file to download, build it from the geometry data.
        # It's less precise, but not likely by enough to matter.
        criteria = {'auid': data_in}
        properties = ['geometry', 'species',
                      'composition', 'positions_fractional']
        if store:
            supplemental_data = store.query_one(criteria=criteria,
                                                properties=properties)
            del supplemental_data['_id']
            entry = Entry(**supplemental_data)
        else:
            q = AflowAPIQuery.from_pymongo(criteria, properties, 1)
            entry = q.__next__()

        from pymatgen.core.lattice import Lattice
        # This lazy-loads the data by making an HTTP request for each property it needs
        # if no mongo store was specified
        geometry = entry.geometry
        lattice = Lattice.from_parameters(*geometry)
        elements = list(map(str.strip, entry.species))
        composition = list(entry.composition)
        species = list(chain.from_iterable([elem] * comp for elem, comp in zip(elements, composition)))
        xyz = entry.positions_fractional.tolist()
        return Structure(lattice, species, xyz)


class AflowAPIQuery(_RetrievalQuery):
    def __init__(self, *args, max_sim_requests=10,
                 batch_reduction=True, property_reduction=False,
                 **kwargs):
        self._executor = ThreadPoolExecutor(max_workers=max_sim_requests)
        self._auto_adjust_batch_size = batch_reduction
        self._auto_adjust_num_props = property_reduction

        self._session = requests.Session()
        retries = Retry(total=3, backoff_factor=10, status_forcelist=[500], connect=0)
        self._session.mount('http://', HTTPAdapter(max_retries=retries))

        super(AflowAPIQuery, self).__init__(*args, **kwargs)

    def __del__(self):
        if self._executor:
            self._executor.shutdown()

    @classmethod
    def from_pymongo(cls, criteria, properties, request_size, **kwargs):
        """Generates an aflow Query object from pymongo-like arguments.

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
        """Constructs the query string for this :class:`Query` object for the
        specified paging limits and then returns the response from the REST API
        as a python object.

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
        starting_entry = (n-1)*k+1
        last_entry = original_n*original_k
        new_k = k // 2
        new_n = starting_entry // new_k + 1
        new_pages = (last_entry-starting_entry) // new_k + 1
        return new_n, new_k, new_pages
        
    @staticmethod
    def _get_response(url, session=None, page=None):
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
        return "{0}{1},{2}".format(server, matchbook, directives)
