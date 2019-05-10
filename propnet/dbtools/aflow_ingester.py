from propnet.ext.aflow import AflowAPIQuery
from propnet.dbtools.aflow_ingester_defaults import default_query_configs, default_files_to_ingest
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
from urllib.error import HTTPError

logger = logging.getLogger(__name__)


class AFLOWIngester(Builder):
    _available_kws = dict()
    kw_load(_available_kws)
    
    def __init__(self, data_target, auid_target=None,
                 keywords=None, query_configs=None,
                 files_to_ingest=None, filter_null_properties=False,
                 **kwargs):
        self.data_target = data_target
        self.auid_target = auid_target

        if query_configs:
            for query_config in query_configs:
                if not all(k in query_config
                           for k in ('catalog', 'k', 'exclude', 'filter', 'select', 'targets')):
                    raise ValueError("Incorrect configuration construction.")
            self.query_configs = query_configs
        else:
            self.query_configs = default_query_configs

        self.files_to_ingest = files_to_ingest or default_files_to_ingest
        self.filter_null_properties = filter_null_properties
        
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
                    query.filter(oper(rhs))
                else:
                    query.filter(oper())
        query.orderby(K.auid)
        return query

    def get_items(self):
        kws = self.keywords.copy()
        for kw in ('auid', 'aurl', 'compound', 'files'):
            try:
                kws.remove(kw)
            except KeyError:
                pass

        for config_ in self.query_configs:
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
                            if data_query.N == 0:   # Empty query
                                raise ValueError("Query returned no results. Query config:\n{}".format(config_))
                            logger.debug('Resting...starting {}'.format(datetime.datetime.now()))   # pragma: no cover
                            time.sleep(120)     # pragma: no cover

    def process_item(self, item):
        entry, targets = item
        kws = self.keywords.copy()
        kws.add('auid')
        if 'auid' in targets and self.auid_target is not None:
            auid_data = {k: entry.raw.get(k, None)
                         for k in ('auid', 'aurl', 'compound')}
        else:
            auid_data = dict()
        db_data = {k: entry.raw.get(k, None) for k in self.keywords}

        if 'files' in entry.attributes:
            file_data = dict()
            for filename in self.files_to_ingest:
                try:
                    data = entry.files[filename]()
                except (KeyError, HTTPError):
                    # Invalid file name, or file does not exist
                    data = None
                if data:
                    file_data[filename.replace('.', '_')] = data
            db_data.update(file_data)

        if self.filter_null_properties:
            db_data = {k: v for k, v in db_data.items() if v is not None}
            auid_data = {k: v for k, v in auid_data.items() if v is not None}
        
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
    
    def finalize(self, cursor=None):    # pragma: no cover
        self.data_target.ensure_index('auid')
        if self.auid_target is not None:
            self.auid_target.ensure_index('auid')
        
        super().finalize(cursor)
