from propnet.ext.aflow import AflowAPIQuery
from aflow.keywords import load, reset as kw_reset
from aflow import K
from maggma.builders import Builder
from maggma.utils import grouper
from monty.json import jsanitize
from pymongo import UpdateOne
import logging
from itertools import zip_longest
import time
import datetime


# noinspection PyUnresolvedReferences
import propnet.ext.aflow_redefs

# Issues with the AFLUX API or the python wrapper
unavailable_aflux_keys = ('ael_elastic_anistropy', 'Pullay_stress',
                          'aflowlib_entries', 'aflowlib_entries_number',
                          'author', 'catalog', 'corresponding', 'data_language',
                          'keywords', 'pressure_final', 'sponsor')

logger = logging.getLogger(__name__)

_query_configs = [
    {
        'catalog': 'icsd',
        'k': 200000,
        'exclude': [],
        'filter': [],
        'select': ['auid', 'aurl', 'compound'],
        'targets': ['data', 'auid']
    },
    {
        'catalog': 'lib1',
        'k': 200000,
        'exclude': [],
        'filter': [],
        'select': ['auid', 'aurl', 'compound'],
        'targets': ['data', 'auid']
    },
    {
        'catalog': 'lib2',
        'k': 200000,
        'exclude': [],
        'filter': [],
        'select': ['auid', 'aurl', 'compound'],
        'targets': ['data', 'auid']
    },
    {
        'catalog': 'lib3',
        'k': 200000,
        'exclude': [],
        'filter': [('auid', '__lt__', "'aflow:{}'".format(start))
                   for start in [hex(i)[2:] for i in range(0, 16)]],
        'select': ['auid', 'aurl', 'compound'],
        'targets': ['data', 'auid']
    },
    {
        'catalog': 'icsd',
        'k': 200000,
        'exclude': ['compound', 'aurl'],
        'filter': [],
        'select': [],
        'targets': ['data']
    },
    {
        'catalog': 'lib1',
        'k': 200000,
        'exclude': ['compound', 'aurl'],
        'filter': [],
        'select': [],
        'targets': ['data']
    },
    {
        'catalog': 'lib2',
        'k': 200000,
        'exclude': ['compound', 'aurl'],
        'filter': [],
        'select': [],
        'targets': ['data']
    },
    {
        'catalog': 'lib3',
        'k': 200000,
        'exclude': ['compound', 'aurl'],
        'filter': [('auid', '__lt__', "'aflow:{}'".format(start))
                   for start in [hex(i)[2:] for i in range(0, 16)]],
        'select': [],
        'targets': ['data']
    },
    {
        'catalog': 'icsd',
        'k': 10000,
        'exclude': ['compound', 'aurl'],
        'filter': [],
        'select': ["files"],
        'targets': ['data']
    },
    {
        'catalog': 'lib1',
        'k': 10000,
        'exclude': ['compound', 'aurl'],
        'filter': [],
        'select': ["files"],
        'targets': ['data']
    },
    {
        'catalog': 'lib2',
        'k': 10000,
        'exclude': ['compound', 'aurl'],
        'filter': [],
        'select': ["files"],
        'targets': ['data']
    },
    {
        'catalog': 'lib3',
        'k': 10000,
        'exclude': ['compound', 'aurl'],
        'filter': [('auid', '__lt__', "'aflow:{}'".format(start))
                   for start in [hex(i)[2:] for i in range(0, 16)]],
        'select': ["files"],
        'targets': ['data']
    },
]


class AFLOWIngester(Builder):
    _available_kws = dict()
    load(_available_kws)
    _available_kws = [k for k in _available_kws.keys()
                      if k not in unavailable_aflux_keys]
    
    def __init__(self, data_target, auid_target=None,
                 keywords=None, batch_size=1000, **kwargs):
        self.data_target = data_target
        self.auid_target = auid_target
        if batch_size > 1000:
            logger.warning("It is recommended to keep batch_size <= 1000 "
                           "or the API request may be rejected by the server.")
        self.batch_size = batch_size
        self.total = None
        
        bad_kw_check = [k for k in keywords or [] if k not in self._available_kws]
        if len(bad_kw_check) != 0:
            raise KeyError("Bad keywords:\n{}".format(bad_kw_check))
        self.keywords = set(keywords or self._available_kws)

        targets = [data_target]
        if self.auid_target is not None:
            targets.append(self.auid_target)

        super(AFLOWIngester, self).__init__(sources=[], targets=targets,
                                            **kwargs)

    @staticmethod
    def _get_query_obj(catalog, batch_size, exclude=None, fv=None):
        kw_reset()
        query = AflowAPIQuery(catalog=catalog, batch_size=batch_size,
                              reduction_scheme='batch')
        if exclude:
            query.exclude(*exclude)
        if fv:
            filters = []
            for filter_item in fv:
                lhs, oper, rhs = filter_item
                lhs = getattr(K, lhs)
                oper = getattr(lhs, oper)
                if rhs:
                    filters.append(oper(lhs, rhs))
                else:
                    filters.append(oper(lhs))
            query.filter(*filters)
        query.orderby(K.auid)
        return query

    def get_items(self):
        kws = self.keywords.copy()
        for kw in ('auid', 'aurl', 'compound', 'files'):
            try:
                kws.remove(kw)
            except ValueError:
                pass

        for config in _query_configs:
            logger.debug(
                "Catalog {} selecting {}".format(config['catalog'],
                                                 'all' if not config['select']
                                                 else config['select']))

            if config['select']:
                kws_to_chunk = config['select']
            else:
                kws_to_chunk = self.keywords

            for n, k, filter_vals in zip_longest(config['n'], config['k'], config['filter'], fillvalue=None):

                chunk_idx = 0
                chunk_size = 5
                total_chunks = len(kws_to_chunk) // chunk_size + 1

                for chunk in grouper(kws_to_chunk, chunk_size):
                    chunk_idx += 1
                    logger.debug("Property chunk {} of {}".format(chunk_idx, total_chunks))
                    props = [getattr(K, c) for c in chunk if c is not None]
                    if len(props) == 0:
                        continue
                    data_query = self._get_query_obj(config['catalog'], k,
                                                     config['exclude'], filter_vals)
                    data_query.select(*props)
                    success = False
                    while not success:
                        try:
                            data_query._request(n, k)
                            success = True
                        except ValueError:
                            logger.debug('Resting...starting {}'.format(datetime.datetime.now()))
                            time.sleep(120)

                    for item in data_query.responses[n].values():
                        yield item, config['targets']

    def process_item(self, item):
        data, targets = item
        kws = self.keywords.copy()
        kws.add('auid')
        if 'auid' in targets:
            auid_data = {k: data.get(k, None)
                         for k in ('auid', 'aurl', 'compound')}
        else:
            auid_data = None
        db_data = {k: data.get(k, None) for k in self.keywords}
        db_data = {k: v for k, v in db_data.items() if v is not None}
        
        return jsanitize(db_data, strict=True), jsanitize(auid_data, strict=True)
    
    def update_targets(self, items):
        data_entries = [UpdateOne(filter={'auid': item[0]['auid']},
                                  update={'$set': item[0]},
                                  upsert=True)
                        for item in items]
        self.data_target.collection.bulk_write(data_entries)

        if self.auid_target is not None:
            auid_entries = [UpdateOne(filter={'auid': item[1]['auid']},
                                      update={'$set': item[1]},
                                      upsert=True)
                            for item in items]
            self.auid_target.collection.bulk_write(auid_entries)
    
    def finalize(self, cursor=None):
        self.data_target.ensure_index('auid')
        if self.auid_target is not None:
            self.auid_target.ensure_index('auid')
        
        super().finalize(cursor)
