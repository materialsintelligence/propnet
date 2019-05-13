from propnet.ext.aflow import AflowAPIQuery
from propnet.dbtools.aflow_ingester_defaults import default_query_configs, default_files_to_ingest
from aflow.keywords import load as kw_load, reset as kw_reset
from aflow import K as AFLOW_KWS
from maggma.builders import Builder
from maggma.utils import grouper
from monty.json import jsanitize
from pymongo import UpdateOne
import logging
import time
import datetime
from urllib.error import HTTPError

logger = logging.getLogger(__name__)


class AflowIngester(Builder):
    """
    Builds MongoDB collections from AFLOW data using the AFLOW and AFLUX web APIs.
    """
    _available_kws = dict()
    """Contains supported keywords in the AFLUX schema
    """
    kw_load(_available_kws)
    
    def __init__(self, data_target, auid_target=None,
                 keywords=None, query_configs=None,
                 files_to_ingest=None, filter_null_properties=False,
                 **kwargs):
        """
        Initialize the database builder.

        Args:
            data_target (maggma.stores.MongoStore): target to store the AFLOW data
            auid_target (`maggma.stores.MongoStore` or `None`): target to store auid, AFLOW API url,
                and formula data as a second database
            keywords (list): list of keywords as strings to download from the AFLOW API. Default: all available
            query_configs (`list` of `dict`): a list of query parameters to use to download the database. The
                configurations will be applied sequentially. See `propnet.dbtools.aflow_ingester_defaults` to
                see schema. Default: the list at `propnet.dbtools.aflow_ingester_defaults.default_query_configs`.
            files_to_ingest (list): list of file names to download as external files.
                Default: the list at `propnet.dbtools.aflow_ingester_defaults.default_files_to_ingest`.
            filter_null_properties (bool): True prevents null properties from being added to the database. False
                creates a null field for them in the database. Default: False
            **kwargs: keyword args to the Builder superclass.
        """

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

        super(AflowIngester, self).__init__(sources=[], targets=targets,
                                            **kwargs)

    @staticmethod
    def _get_query_obj(catalog, batch_size, excludes=None, filters=None):
        """
        Produces API query object from selected query configuration.

        Args:
            catalog (str): catalog name of 'icsd', 'lib1', 'lib2', and 'lib3'
            batch_size (int): how many records to retrieve in a single query
            excludes (`list` of `str`): properties to explicitly exclude from query
            filters (`list` of `tuple`): descriptors for constructing query filters

        Returns:
            propnet.ext.AflowAPIQuery: query object
        """
        kw_reset()
        query = AflowAPIQuery(catalog=catalog, batch_size=batch_size,
                              batch_reduction=True, property_reduction=True)
        if excludes:
            query.exclude(*(getattr(AFLOW_KWS, item) for item in excludes))
        if filters:
            for filter_item in filters:
                lhs, oper, rhs = filter_item
                lhs = getattr(AFLOW_KWS, lhs)
                oper = getattr(lhs, oper)
                if rhs:
                    query.filter(oper(rhs))
                else:
                    query.filter(oper())
        query.orderby(AFLOW_KWS.auid)
        return query

    def get_items(self):
        """
        Retrieves AFLOW data using the AFLUX API according to the specifications in the query
        configurations.

        Yields:
            tuple: The first item is an `aflow.entries.Entry` containing the material data
                and the second item is a list of targets for the data ('data' and/or 'auid')

        """
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

            k = config_['k']
            filter_vals = config_['filters']

            chunk_idx = 0
            chunk_size = 5
            total_chunks = len(kws_to_chunk) // chunk_size + 1

            for chunk in grouper(kws_to_chunk, chunk_size):
                chunk_idx += 1
                logger.debug("Property chunk {} of {}".format(chunk_idx, total_chunks))
                props = [getattr(AFLOW_KWS, c) for c in chunk if c is not None]
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
                        else:   # pragma: no cover
                            logger.warning(
                                'Server error. ' +
                                'Resting...starting {}'.format(datetime.datetime.now()))
                            time.sleep(120)

    def process_item(self, item):
        """
        Processes AFLOW data, filters by null properties, and downloads extra files as available.

        Args:
            item (tuple): data tuple from `get_items()`

        Returns:
            tuple: dicts of JSON-sanitized data for the data (position #1) and auid (position #2) stores.
                If data is not destined for one of the targets, tuple position consists of an
                empty dictionary.
        """
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
        """
        Updates Mongo stores. If an auid entry does not exist, it is created.
        If a previous query configuration downloaded the same field, the field
        will be overwritten.

        Args:
            items (list): list of data tuples from `process_item()`
        """
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
        """
        Wraps up database build, ensuring indices exist in the stores and closing
        the cursor if necessary.

        Args:
            cursor (pymongo.cursor.Cursor): MongoDB cursor to be closed

        """
        self.data_target.ensure_index('auid')
        if self.auid_target is not None:
            self.auid_target.ensure_index('auid')
        
        super().finalize(cursor)
