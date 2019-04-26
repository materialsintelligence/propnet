from propnet.ext.aflow import AsyncQuery
from aflow.keywords import load
from maggma.builders import Builder
from monty.json import jsanitize
from pymongo import InsertOne
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


# Populate all available kws from AFLOW
AFLOW_KWS = dict()
load(AFLOW_KWS)
AFLOW_KWS = list(AFLOW_KWS.keys())

logger = logging.getLogger(__name__)


class AFLOWIngester(Builder):
    def __init__(self, data_target, auid_target,
                 keywords=None, auid_batch_size=1000, insert_only=False,
                 **kwargs):
        self.data_target = data_target
        self.auid_target = auid_target
        if auid_batch_size > 1000:
            logger.warning("It is recommended to keep batch_size <= 1000 "
                           "or the AFLUX API request may be rejected by the server.")
        self.auid_batch_size = auid_batch_size
        self.total = None
        self.insert_only = insert_only
        
        bad_kw_check = [k for k in keywords or [] if k not in AFLOW_KWS]
        if len(bad_kw_check) != 0:
            raise KeyError("Bad keywords:\n{}".format(bad_kw_check))
        self.keywords = keywords
        
        super(AFLOWIngester, self).__init__(sources=[], targets=[data_target, auid_target],
                                            **kwargs)
    
    def get_items(self):
        '''
        afs = aflow_search(batch_size=self.batch_size)
        afs.select(*(self._available_kws[k] for k in self.keywords))
        afs.filter()
        '''
        afs = AsyncQuery.from_pymongo(criteria={},
                                      properties=['auid', 'compound', 'aurl'],
                                      request_size=self.auid_batch_size)
        self.total = afs.N
        yield from afs
            
    def process_item(self, item):
        url: str = item.aurl
        url = url.replace(":", "/")
        if not url.startswith("http://"):
            url = "http://" + url

        auid_info = {k: getattr(item, k, None) for k in ('auid', 'compound', 'aurl')}

        if url is not None:
            s = requests.Session()
            retries = Retry(total=5, backoff_factor=1)
            s.mount('http://', HTTPAdapter(max_retries=retries))
            response = requests.get(url, params={'format': 'json'})

            if response.ok:
                data = jsanitize(response.json())
            else:
                data = jsanitize({'error': response.text}, strict=True)
                data.update(auid_info)
        else:
            data = {'error': 'No URL'}

        return auid_info, data
    
    def update_targets(self, items):
        if self.insert_only:
            auid_entries = [InsertOne(item[0]) for item in items]
            data_entries = [InsertOne(item[1]) for item in items]
            self.auid_target.collection.bulk_write(auid_entries)
            self.data_target.collection.bulk_write(data_entries)
        else:
            auid_entries = [item[0] for item in items]
            data_entries = [item[1] for item in items]
            self.auid_target.update(auid_entries, key='auid')
            self.data_target.update(data_entries, key='auid')
    
    def finalize(self, cursor=None):
        self.auid_target.ensure_index('auid')
        self.data_target.ensure_index('auid')
        
        super().finalize(cursor)
