from propnet.ext.aflow import AsyncQuery
from aflow.keywords import load
from maggma.builders import Builder
from monty.json import jsanitize
from pymongo import InsertOne

# noinspection PyUnresolvedReferences
import propnet.dbtools.aflow_redefs

# Issues with the AFLUX API or the python wrapper
unavailable_aflux_keys = ('ael_elastic_anistropy', 'Pullay_stress',
                          'aflowlib_entries', 'aflowlib_entries_number',
                          'author', 'catalog', 'corresponding', 'data_language',
                          'keywords', 'pressure_final', 'pressure_residual', 'sponsor')


class AFLOWIngester(Builder):
    def __init__(self, data_target, auid_target,
                 keywords=None, batch_size=10000, insert_only=False,
                 **kwargs):
        self.data_target = data_target
        self.auid_target = auid_target
        self.batch_size = batch_size
        self.total = None
        self.insert_only = insert_only

        self._available_kws = dict()
        load(self._available_kws)
        self._available_kws = {k: v for k, v in self._available_kws.items()
                               if k not in unavailable_aflux_keys}
        
        bad_kw_check = [k for k in keywords or [] if k not in self._available_kws]
        if len(bad_kw_check) != 0:
            raise KeyError("Bad keywords:\n{}".format(bad_kw_check))
        self.keywords = keywords or list(self._available_kws.keys())
        
        super(AFLOWIngester, self).__init__(sources=[], targets=[data_target, auid_target],
                                            **kwargs)
    
    def get_items(self):
        '''
        afs = aflow_search(batch_size=self.batch_size)
        afs.select(*(self._available_kws[k] for k in self.keywords))
        afs.filter()
        '''
        afs = AsyncQuery.from_pymongo(criteria={},
                                      properties=self.keywords,
                                      request_size=self.batch_size)
        self.total = afs.N
        yield from afs
            
    def process_item(self, item):
        d = {k: getattr(item, k, None) for k in self.keywords}
        d = {k: v for k, v in d.items() if v is not None}
        auid_info = {k: d[k] for k in ('auid', 'compound', 'aurl')}
        
        return jsanitize(auid_info, strict=True), jsanitize(d, strict=True)
    
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
