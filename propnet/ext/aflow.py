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
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from requests.exceptions import ConnectionError, RetryError
import logging
from collections import defaultdict

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
    def __init__(self, *args, max_sim_requests=10, reduction_scheme='batch',
                 **kwargs):
        self._executor = ThreadPoolExecutor(max_workers=max_sim_requests)
        if reduction_scheme == 'batch':
            self._auto_adjust_batch_size = True
            self._auto_adjust_num_props = False
        elif reduction_scheme == 'property':
            self._auto_adjust_batch_size = False
            self._auto_adjust_num_props = True
        elif reduction_scheme is None:
            self._auto_adjust_batch_size = False
            self._auto_adjust_num_props = False
        else:
            raise ValueError("Invalid reduction scheme")

        self._session = requests.Session()
        retries = Retry(total=5, backoff_factor=3, status_forcelist=[500])
        self._session.mount('http://', HTTPAdapter(max_retries=retries))

        super(AsyncQuery, self).__init__(*args, **kwargs)

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

    def orderby(self, keyword, reverse=False):
        """Sets a keyword to be the one by which

        Args:
            keyword (aflow.keywords.Keyword): that encapsulates the AFLUX
              request language logic.
            reverse (bool): when True, reverse the ordering.
        """
        if self._final_check():
            self._N = None
            self.order = keyword
            self.reverse = reverse
            if str(keyword) in map(str, self.selects):
                idx = list(map(str, self.selects)).index(str(keyword))
                self.selects.pop(idx)
        return self
        
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
        logger.debug('Requesting page {} with {} records'.format(n, k))
        try:
            is_ok, response = self._get_response(request_url, session=self._session)
        except ConnectionError as ex:
            # We requested SO many things that the server rejected our request
            # outright as opposed to trying to complete the request and failing
            is_ok = False
            response = ex.args

        if not is_ok:
            if self._auto_adjust_batch_size:
                response = self._request_with_smaller_batch(n, k)
            elif self._auto_adjust_num_props:
                response = self._request_with_fewer_props(n, k)
            else:
                raise ValueError("The API failed to complete the request.")

        # If this is the first request, then save the number of results in the
        # query.
        if len(self.responses) == 0:
            self._N = int(next(iter(response.keys())).split()[-1])

        # Filter out any extra responses that we got
        collected_responses = {kk: v for kk, v in response.items()
                               if int(kk.split()[0]) <= n*k}
        self.responses[n] = collected_responses

    def _request_with_fewer_props(self, n, k):
        collected_responses = defaultdict(dict)
        props = self.selects
        max_chunks = 5
        chunks = 2
        while chunks <= max_chunks or len(props) // chunks == 0:
            query_error = False
            for chunk in grouper(props, (len(props) // chunks) + 1):
                logger.debug('Requesting property chunk {} with {} records'.format(chunks, k))
                props_to_request = set(c for c in chunk if c is not None)
                props_to_request.add(str(self.order))
                query = AsyncQuery.from_pymongo(criteria={},
                                                properties=list(props_to_request),
                                                request_size=k,
                                                reduction_scheme=None)
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
            retries = Retry(total=5, backoff_factor=3, status_forcelist=[500])
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
