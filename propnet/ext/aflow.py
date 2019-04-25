from matminer.data_retrieval.retrieve_AFLOW import AFLOWDataRetrieval, RetrievalQuery as _RetrievalQuery
from aflow.control import server as _aflow_server
from aflow import K, msg as _msg
from concurrent.futures import ThreadPoolExecutor, as_completed
from maggma.utils import grouper
from propnet.core.materials import Material
from propnet.core.quantity import QuantityFactory
from propnet.core.provenance import ProvenanceElement
import pandas as pd
from datetime import datetime
import requests
from requests.exceptions import ConnectionError


class AFLOWRetrieve(AFLOWDataRetrieval):
    # Mapping is keyed by AFLOW keyword, valued by propnet property
    mapping = {
        "auid": "external_identifier_aflow",
        "Egap": "band_gap",
        "ael_bulk_modulus_reuss": "bulk_modulus",
        "ael_bulk_modulus_voigt": "bulk_modulus",
        # "ael_elastic_anistropy": "elastic_anisotropy",    # This property returns "not allowed" from API
        "ael_poisson_ratio": "poisson_ratio",
        "ael_shear_modulus_reuss": "shear_modulus",
        "ael_shear_modulus_voigt": "shear_modulus",
        "agl_acoustic_debye": "debye_temperature",
        "agl_debye": "debye_temperature",
        "agl_gruneisen": "gruneisen_parameter",
        # Listed per unit cell, need new symbol and model for conversion
        # "agl_heat_capacity_Cp_300K": "heat_capacity_of_cell_constant_pressure",
        # "agl_heat_capacity_Cv_300K": "heat_capacity_of_cell_constant_volume",
        "agl_thermal_conductivity_300K": "thermal_conductivity",
        "agl_thermal_expansion_300K": "thermal_expansion_coefficient",
        # This returns the volumes of all the atoms' volumes
        # "bader_atomic_volumes": ""
        "compound": "formula",
        "energy_atom": "energy_per_atom",
        "enthalpy_formation_atom": "formation_energy_per_atom"
    }
    file_property_mapping = {
        "structure": "structure"
    }
    transform_func = {
        "energy_atom": lambda x: abs(x)
    }
    unit_map = {
        "Egap": "eV",
        "ael_bulk_modulus_reuss": "gigapascal",
        "ael_bulk_modulus_voigt": "gigapascal",
        # "ael_elastic_anistropy": "dimensionless",
        "ael_poisson_ratio": "dimensionless",
        "ael_shear_modulus_reuss": "gigapascal",
        "ael_shear_modulus_voigt": "gigapascal",
        "agl_acoustic_debye": "kelvin",
        "agl_debye": "kelvin",
        "agl_gruneisen": "dimensionless",
        # "agl_heat_capacity_Cp_300K": "boltzmann_constant",
        # "agl_heat_capacity_Cv_300K": "boltzmann_constant",
        "agl_thermal_conductivity_300K": "watt/meter/kelvin",
        "agl_thermal_expansion_300K": "1/kelvin",
        "energy_atom": "eV/atom",
        "enthalpy_formation_atom": "eV/atom"
    }

    def __init__(self, max_sim_requests=10):
        self._executor = ThreadPoolExecutor(max_workers=max_sim_requests)
        super(AFLOWRetrieve, self).__init__()

    def __del__(self):
        if self._executor:
            self._executor.shutdown()
    
    def get_material_by_auid(self, auid):
        return self.get_materials_by_auids([auid])[0]
    
    def get_materials_by_auids(self, auids, max_request_size=1000):
        return [m for m in self.generate_materials_by_auids(auids, max_request_size)]
    
    def generate_materials_by_auids(self, auids, max_request_size=1000):
        futures = self._submit_auid_queries(auids, max_request_size)

        for f in as_completed(futures):
            try:
                response: pd.DataFrame = f.result()
            except Exception as ex:
                msg = "Could not retrieve one or more materials. Error:\n{}".format(ex)
                for future in futures:
                    future.cancel()
                raise ValueError(msg)

            for auid, data in response.iterrows():
                yield self._transform_response_to_material(auid, data)
    
    @staticmethod
    def generate_all_auids(max_request_size=1000, with_metadata=True):
        props = ['auid']
        if with_metadata:
            props += ['compound', 'aflowlib_date']
        query = AsyncQuery.from_pymongo(
            criteria={},
            properties=props,
            request_size=max_request_size
        )

        for item in query.generate_items(preserve_order=False):
            for d in item.values():
                if with_metadata:
                    yield d
                else:
                    yield d['auid']

    def _submit_auid_queries(self, auids, max_request_size=1000):
        futures = []
        for chunk in grouper(auids, max_request_size):
            criteria = {'auid': {'$in': [c for c in chunk if c is not None]}}
            properties = list(self.mapping.keys()) + ['aflowlib_date']
            files = list(self.file_property_mapping.keys())
            f = self._executor.submit(
                self.get_dataframe, criteria, properties, files=files)
            futures.append(f)
        return futures

    def _transform_response_to_material(self, auid, data):
        qs = []
        for prop, value in data.items():
            if value is not None and \
                    (prop in self.mapping or prop in self.file_property_mapping):
                date_created = data.get('aflowlib_date')
                if date_created:
                    date, tz = date_created.rsplit("GMT", 1)
                    tz = "GMT{:+05d}".format(int(tz) * 100)
                    date_object = datetime.strptime(date + tz, "%Y%m%d_%H:%M:%S_%Z%z")
                    date_created = date_object.strftime("%Y-%m-%d %H:%M:%S")
                provenance = ProvenanceElement(
                    source={'source': 'AFLOW',
                            'source_key': auid,
                            'date_created': date_created}
                )
                if prop in self.transform_func:
                    value = self.transform_func[prop](value)
                q = QuantityFactory.create_quantity(
                    self.mapping.get(prop) or self.file_property_mapping.get(prop),
                    value,
                    units=self.unit_map.get(prop), provenance=provenance
                )
                qs.append(q)
        return Material(qs)

    
class AsyncQuery(_RetrievalQuery):
    def __init__(self, *args, max_sim_requests=10, **kwargs):
        self._executor = ThreadPoolExecutor(max_workers=max_sim_requests)
        super(AsyncQuery, self).__init__(*args, **kwargs)

    def __del__(self):
        if self._executor:
            self._executor.shutdown()

    @classmethod
    def from_pymongo(cls, criteria, properties, request_size):
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
        query = cls(batch_size=request_size)

        # adds filters to query
        query._add_filters(criteria)

        # determines properties returned by query
        query.select(*[getattr(K, i) for i in properties])

        # suppresses properties that may have been included as search criteria
        # but are not requested properties, which the user wants returned
        excluded_keywords = set(criteria.keys()) - set(properties)
        query.exclude(*[getattr(K, i) for i in excluded_keywords])

        return query

    def generate_items(self, preserve_order=True):
        self.reset_iter()
        self._request(1, self.k)
        yield self.responses[1]

        urls = [self._get_request_url(_aflow_server, self.matchbook(), self._directives(page, self.k))
                for page in range(2, self._N)]

        futures = []
        for page, url in enumerate(urls):
            f = self._executor.submit(self._get_response, url, page+1)
            futures.append(f)
            
        for f in self._get_next_future(futures, preserve_order):
            try:
                _, result, page = f.result()
            except Exception as ex:
                for ff in futures:
                    ff.cancel()
                raise ex
            self.responses[page] = result
            yield result
        
    @staticmethod
    def _get_next_future(futures, preserve_order=True):
        if preserve_order:
            yield from futures
        else:
            yield from as_completed(futures)
        
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
        try:
            _, response = self._get_response(request_url)
        except ConnectionError:
            raise ValueError("API request was rejected. Is your number"
                             " of records per page much greater than 1000?")

        # If this is the first request, then save the number of results in the
        # query.
        if len(self.responses) == 0:
            self._N = int(next(iter(response.keys())).split()[-1])
        self.responses[n] = response
    
    @staticmethod
    def _get_response(url, page=None):
        rawresp = requests.get(url)
        retry = 0
        while not rawresp.ok and retry < 5:
            retry += 1
            rawresp = requests.get(url)

        if rawresp.ok:
            response = rawresp.json()
        else:
            _msg.err("{}\n\n{}".format(url, rawresp))
            response = rawresp.text
        if page is not None:
            return rawresp.ok, response, page
        return rawresp.ok, response

    @staticmethod
    def _get_request_url(server, matchbook, directives):
        return "{0}{1},{2}".format(server, matchbook, directives)
