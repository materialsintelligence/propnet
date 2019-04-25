# from propnet.ext.aflow import AFLOWRetrieve
from matminer.data_retrieval.retrieve_AFLOW import RetrievalQuery
from aflow.control import search as aflow_search
from aflow.keywords import load
from maggma.builders import Builder
from monty.json import jsanitize

unavailable_aflux_keys = ('ael_elastic_anistropy', 'Pullay_stress',
                          'aflowlib_entries', 'aflowlib_entries_number',
                          'author', 'catalog', 'corresponding', 'data_language',
                          'keywords', 'delta_electronic_energy_convergence',
                          'delta_electronic_energy_threshold', 'forces',
                          'ldau_TLUJ', 'positions_cartesian', 'positions_fractional',
                          'pressure_final', 'pressure_residual', 'sponsor',
                          'stoich', 'stress_tensor', 'kpoints')     # Issues with the AFLUX API or the python wrapper


class AFLOWIngester(Builder):
    def __init__(self, data_target, auid_target, keywords=None, batch_size=10000, **kwargs):
        self.data_target = data_target
        self.auid_target = auid_target
        self.batch_size = batch_size

        self._available_kws = dict()
        load(self._available_kws)
        for k in unavailable_aflux_keys:
            self._available_kws.pop(k)
        
        unavailable_kw_check = set(keywords or []).intersection(unavailable_aflux_keys)
        if len(unavailable_kw_check) != 0:
            raise KeyError("Cannot request keys:\n{}".format(unavailable_kw_check))
        
        bad_kw_check = [k not in self._available_kws for k in keywords or []]
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
        afs = RetrievalQuery.from_pymongo(criteria={},
                                          properties=self.keywords,
                                          request_size=self.batch_size)
        yield from afs
            
    def process_item(self, item):
        print("I made it")
        d = {k: getattr(item, k, None) for k in self.keywords}
        d = {k: v for k, v in d.items() if v is not None}
        auid_info = {k: d[k] for k in ('auid', 'compound', 'aurl')}
        
        return jsanitize(auid_info, strict=True), jsanitize(d, strict=True)
    
    def update_targets(self, items):
        auid_entries = [item[0] for item in items]
        data_entries = [item[1] for item in items]
        
        self.auid_target.update(auid_entries, key='auid')
        self.data_target.update(data_entries, key='auid')
    
    def finalize(self, cursor=None):
        self.auid_target.ensure_index('auid')
        self.data_target.ensure_index('auid')
        
        super().finalize(cursor)
