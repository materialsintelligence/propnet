from propnet.ext.aflow import AflowAPIQuery
from aflow.keywords import load as kw_load, reset as kw_reset
from aflow import K
from maggma.builders import Builder
from maggma.utils import grouper
from monty.json import jsanitize
from pymongo import UpdateOne
import logging
from itertools import zip_longest
import time
import datetime

logger = logging.getLogger(__name__)

# Querying AFLOW needs to be done in manageable chunks
# These configuations define the chunks to download. Note that
# this will not download all properties for a material at once,
# so it must be run in its entirety to complete the material.

_files_to_ingest = [
    'CONTCAR.relax.vasp',
    'AEL_elastic_tensor.json'
]

_query_configs = [
    {
        'catalog': 'icsd',
        'k': [200000],
        'exclude': [],
        'filter': [],
        'select': ['auid', 'aurl', 'compound'],
        'targets': ['data', 'auid']
    },
    {
        'catalog': 'icsd',
        'k': [200000],
        'exclude': ['compound', 'aurl'],
        'filter': [],
        'select': [],
        'targets': ['data']
    },
    {
        'catalog': 'icsd',
        'k': [10000],
        'exclude': ['compound'],
        'filter': [],
        'select': ['files', 'aurl'],
        'targets': ['data']
    },
    {
        'catalog': 'lib1',
        'k': [200000],
        'exclude': [],
        'filter': [],
        'select': ['auid', 'aurl', 'compound'],
        'targets': ['data', 'auid']
    },
    {
        'catalog': 'lib1',
        'k': [200000],
        'exclude': ['compound', 'aurl'],
        'filter': [],
        'select': [],
        'targets': ['data']
    },
    {
        'catalog': 'lib1',
        'k': [10000],
        'exclude': ['compound'],
        'filter': [],
        'select': ['files', 'aurl'],
        'targets': ['data']
    },
    {
        'catalog': 'lib2',
        'k': [200000],
        'exclude': [],
        'filter': [],
        'select': ['auid', 'aurl', 'compound'],
        'targets': ['data', 'auid']
    },
    {
        'catalog': 'lib2',
        'k': [200000],
        'exclude': ['compound', 'aurl'],
        'filter': [],
        'select': [],
        'targets': ['data']
    },
    {
        'catalog': 'lib2',
        'k': [10000],
        'exclude': ['compound'],
        'filter': [],
        'select': ['files', 'aurl'],
        'targets': ['data']
    },
    {
        'catalog': 'lib3',
        'k': [200000],
        'exclude': [],
        'filter': [('auid', '__lt__', "'aflow:{}'".format(start))
                   for start in [hex(i)[2:] for i in range(0, 16)]],
        'select': ['auid', 'aurl', 'compound'],
        'targets': ['data', 'auid']
    },
    {
        'catalog': 'lib3',
        'k': [200000],
        'exclude': ['compound', 'aurl'],
        'filter': [('auid', '__lt__', "'aflow:{}'".format(start))
                   for start in [hex(i)[2:] for i in range(0, 16)]],
        'select': [],
        'targets': ['data']
    },
    {
        'catalog': 'lib3',
        'k': [10000],
        'exclude': ['compound'],
        'filter': [('auid', '__lt__', "'aflow:{}'".format(start))
                   for start in [hex(i)[2:] for i in range(0, 16)]],
        'select': ['files', 'aurl'],
        'targets': ['data']
    },
]


class AFLOWIngester(Builder):
    _available_kws = dict()
    kw_load(_available_kws)
    
    def __init__(self, data_target, auid_target=None,
                 keywords=None, **kwargs):
        self.data_target = data_target
        self.auid_target = auid_target
        
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
    def _get_query_obj(catalog, batch_size, excludes=None, filters=None):
        kw_reset()
        query = AflowAPIQuery(catalog=catalog, batch_size=batch_size,
                              batch_reduction=True, property_reduction=True)
        if excludes:
            query.exclude(*(getattr(K, item) for item in excludes))
        if filters:
            for filter_item in filters:
                lhs, oper, rhs = filter_item
                lhs = getattr(K, lhs)
                oper = getattr(lhs, oper)
                if rhs:
                    query.filter(oper(lhs, rhs))
                else:
                    query.filter(oper(lhs))
        query.orderby(K.auid)
        return query

    def get_items(self):
        kws = self.keywords.copy()
        for kw in ('auid', 'aurl', 'compound', 'files'):
            try:
                kws.remove(kw)
            except ValueError:
                pass

        for config_ in _query_configs:
            logger.debug(
                "Catalog {} selecting {}".format(config_['catalog'],
                                                 'all' if not config_['select']
                                                 else config_['select']))

            if config_['select']:
                kws_to_chunk = config_['select']
            else:
                kws_to_chunk = self.keywords

            for k, filter_vals in zip_longest(config_['k'], config_['filter'], fillvalue=None):

                chunk_idx = 0
                chunk_size = 5
                total_chunks = len(kws_to_chunk) // chunk_size + 1

                for chunk in grouper(kws_to_chunk, chunk_size):
                    chunk_idx += 1
                    logger.debug("Property chunk {} of {}".format(chunk_idx, total_chunks))
                    props = [getattr(K, c) for c in chunk if c is not None]
                    if len(props) == 0:
                        continue
                    data_query = self._get_query_obj(config_['catalog'], k,
                                                     config_['exclude'], filter_vals)
                    data_query.select(*props)
                    success = False
                    while not success:
                        try:
                            for entry in data_query:
                                yield entry, config_['targets']
                            success = True
                        except ValueError:
                            logger.debug('Resting...starting {}'.format(datetime.datetime.now()))
                            time.sleep(120)

    def process_item(self, item):
        entry, targets = item
        kws = self.keywords.copy()
        kws.add('auid')
        if 'auid' in targets:
            auid_data = {k: entry.raw.get(k, None)
                         for k in ('auid', 'aurl', 'compound')}
        else:
            auid_data = None
        db_data = {k: entry.raw.get(k, None) for k in self.keywords}
        db_data = {k: v for k, v in db_data.items() if v is not None}

        if 'files' in entry.attributes:
            file_data = dict()
            for filename in _files_to_ingest:
                try:
                    data = entry.files[filename]()
                except KeyError:
                    data = None
                if data:
                    file_data[filename.replace('.', '_')] = data
            db_data.update(file_data)
        
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
