# from propnet.ext.aflow import AFLOWRetrieve
from matminer.data_retrieval.retrieve_AFLOW import RetrievalQuery
from aflow.control import search as aflow_search
from aflow.keywords import load
import aflow.caster
from maggma.builders import Builder
from monty.json import jsanitize
import numpy as np

unavailable_aflux_keys = ('ael_elastic_anistropy', 'Pullay_stress',
                          'aflowlib_entries', 'aflowlib_entries_number',
                          'author', 'catalog', 'corresponding', 'data_language',
                          'keywords', 'delta_electronic_energy_convergence',
                          'delta_electronic_energy_threshold', 'forces',
                          'ldau_TLUJ', 'positions_cartesian', 'positions_fractional',
                          'pressure_final', 'pressure_residual', 'sponsor',
                          'stoich', 'stress_tensor')     # Issues with the AFLUX API or the python wrapper


# Redefining aflow library functions because they're not general enough
def _numbers(value):
    svals = list(value.split(','))
    vals = list(map(aflow.caster._number, svals))
    return np.array(vals)


def _kpoints(value):
    parts = value.split(';')
    relaxation = np.array(list(map(aflow.caster._number, parts[0].split(','))))
    if len(parts) == 1:
        return {"relaxation": relaxation}
    static = np.array(list(map(aflow.caster._number, parts[1].split(','))))
    if len(parts) == 3:  # pragma: no cover
        # The web page (possibly outdated) includes an example where
        # this would be the case. We include it here for
        # completeness. I haven't found a case yet that we could use in
        # the unit tests to trigger this.
        points = parts[-1].split('-')
        nsamples = None
    else:
        points = parts[-2].split('-')
        nsamples = int(parts[-1])

    return {
        "relaxation": relaxation,
        "static": static,
        "points": points,
        "nsamples": nsamples
    }


aflow.caster._numbers = _numbers
aflow.caster._kpoints = _kpoints


class AFLOWIngester(Builder):
    def __init__(self, data_target, auid_target, keywords=None, batch_size=10000, **kwargs):
        self.data_target = data_target
        self.auid_target = auid_target
        self.batch_size = batch_size
        self.total = None

        self._available_kws = dict()
        load(self._available_kws)
        for k in unavailable_aflux_keys:
            self._available_kws.pop(k)
        
        unavailable_kw_check = set(keywords or []).intersection(unavailable_aflux_keys)
        if len(unavailable_kw_check) != 0:
            raise KeyError("Cannot request keys:\n{}".format(unavailable_kw_check))
        
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
        afs = RetrievalQuery.from_pymongo(criteria={},
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
        auid_entries = [item[0] for item in items]
        data_entries = [item[1] for item in items]
        
        self.auid_target.update(auid_entries, key='auid')
        self.data_target.update(data_entries, key='auid')
    
    def finalize(self, cursor=None):
        self.auid_target.ensure_index('auid')
        self.data_target.ensure_index('auid')
        
        super().finalize(cursor)
