import unittest
from propnet.dbtools.aflow_ingester import AflowIngester
from propnet.dbtools.aflow_ingester_defaults import default_query_configs, default_files_to_ingest
from propnet.symbols import add_builtin_symbols_to_registry
from propnet.core.registry import Registry
from maggma.stores import MemoryStore
from aflow.entries import Entry


class AFLOWIngesterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        Registry.clear_all_registries()
        add_builtin_symbols_to_registry()

    def test_instantiate_builder(self):
        dt = MemoryStore()
        at = MemoryStore()
        afi = AflowIngester(data_target=dt, auid_target=at)

        self.assertEqual(afi.sources, [])
        self.assertEqual(sum(t is dt for t in afi.targets), 1)
        self.assertEqual(sum(t is at for t in afi.targets), 1)
        self.assertEqual(afi.query_configs, default_query_configs)
        self.assertEqual(afi.files_to_ingest, default_files_to_ingest)
        self.assertFalse(afi.filter_null_properties)

        afi = AflowIngester(data_target=dt)

        self.assertEqual(afi.sources, [])
        self.assertEqual(sum(t is dt for t in afi.targets), 1)
        self.assertEqual(sum(t is at for t in afi.targets), 0)

        qc = [{
            'catalog': 'icsd',
            'k': 10000,
            'exclude': [],
            'filter': [],
            'select': [],
            'targets': ['data', 'auid']
        }]
        kw = ['auid', 'ael_elastic_anisotropy', 'auid']
        afi = AflowIngester(data_target=dt, keywords=kw, query_configs=qc, filter_null_properties=True)

        self.assertEqual(afi.keywords, set(kw))
        self.assertListEqual(afi.query_configs, qc)
        self.assertTrue(afi.filter_null_properties)

        with self.assertRaises(KeyError):
            _ = AflowIngester(data_target=dt, keywords=['garbage'])

        with self.assertRaises(ValueError):
            _ = AflowIngester(data_target=dt, query_configs=[{'catalog': 'icsd'}])

    def test_get_item(self):
        dt = MemoryStore()
        at = MemoryStore()
        qc = [{
            'catalog': 'icsd',
            'k': 100,
            'exclude': [],
            'filter': [('auid', '__eq__', 'aflow:0132ab6b9cddd429')],
            'select': [],
            'targets': ['data', 'auid']
        }]
        kw = ['ael_elastic_anisotropy']
        afi = AflowIngester(data_target=dt, auid_target=at, keywords=kw,
                            query_configs=qc)

        count = 0
        for item in afi.get_items():
            count += 1
            self.assertEqual(count, 1, msg="The query returned more than one material")
            expected_attributes = ['auid', 'aurl', 'compound', 'ael_elastic_anisotropy']
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            entry, targets = item
            self.assertIsInstance(entry, Entry)
            self.assertIsInstance(targets, list)

            self.assertCountEqual(list(entry.attributes.keys()), expected_attributes)
            self.assertCountEqual(targets, ['data', 'auid'])

            self.assertEqual(entry.auid, 'aflow:0132ab6b9cddd429')

        # Test empty query
        empty_qc = [{
            'catalog': 'lib1',
            'k': 100,
            'exclude': [],
            'filter': [('auid', '__eq__', 'aflow:0132ab6b9cddd429')],
            'select': [],
            'targets': ['data', 'auid']
        }]
        afi = AflowIngester(data_target=dt, auid_target=at, keywords=kw,
                            query_configs=empty_qc)
        with self.assertRaises(ValueError):
            _ = afi.get_items().__next__()

    def test_process_item(self):
        dt = MemoryStore()
        at = MemoryStore()

        qc = [{
            'catalog': 'icsd',
            'k': 100,
            'exclude': ['compound'],
            'filter': [('auid', '__eq__', 'aflow:0132ab6b9cddd429')],
            'select': [],
            'targets': ['data', 'auid']
        }]
        kw = ['auid', 'aurl', 'ael_elastic_anisotropy', 'files']
        afi = AflowIngester(data_target=dt, auid_target=at, keywords=kw,
                            query_configs=qc)

        item = afi.get_items().__next__()

        processed = afi.process_item(item)

        self.assertIsInstance(processed, tuple)
        self.assertEqual(len(processed), 2)

        data, auid = processed

        expected_data_keys = kw + ['AEL_elastic_tensor_json', 'CONTCAR_relax_vasp']
        expected_auid_keys = ['auid', 'aurl', 'compound']

        self.assertCountEqual(list(data.keys()), expected_data_keys)
        self.assertTrue(all(v is not None for v in data.values()))
        self.assertCountEqual(list(auid.keys()), expected_auid_keys)
        self.assertIsNone(auid['compound'])

        # Omit auid target
        afi = AflowIngester(data_target=dt, keywords=kw,
                            query_configs=qc)

        item = afi.get_items().__next__()

        processed = afi.process_item(item)

        self.assertIsInstance(processed, tuple)
        self.assertEqual(len(processed), 2)

        _, auid = processed

        self.assertEqual(auid, dict())

        # Test for filtering null properties
        qc = [{
            'catalog': 'lib1',
            'k': 100,
            'exclude': ['compound'],
            'filter': [('auid', '__eq__', 'aflow:7203c28b8396b9c9')],
            'select': [],
            'targets': ['data', 'auid']
        }]
        afi = AflowIngester(data_target=dt, auid_target=at, keywords=kw,
                            query_configs=qc, filter_null_properties=True)

        item = afi.get_items().__next__()

        processed = afi.process_item(item)

        self.assertIsInstance(processed, tuple)
        self.assertEqual(len(processed), 2)

        data, auid = processed

        self.assertNotIn('ael_elastic_anisotropy', data)
        self.assertNotIn('AEL_elastic_tensor_json', data)
        self.assertNotIn('compound', auid)

    def test_update(self):
        dt = MemoryStore()
        at = MemoryStore()
        qc = [{
            'catalog': 'icsd',
            'k': 100,
            'exclude': ['compound'],
            'filter': [('auid', '__eq__', 'aflow:0132ab6b9cddd429')],
            'select': [],
            'targets': ['data', 'auid']
        }]
        kw = ['auid', 'aurl', 'ael_elastic_anisotropy', 'files']
        afi = AflowIngester(data_target=dt, auid_target=at, keywords=kw,
                            query_configs=qc)
        afi.connect()

        item = afi.get_items().__next__()

        processed = afi.process_item(item)

        afi.update_targets([processed])

        dt_data = list(dt.query())
        at_data = list(at.query())

        self.assertEqual(len(dt_data), 1)
        self.assertEqual(len(at_data), 1)

        # Omit auid target and use same data target to ensure upsert
        afi = AflowIngester(data_target=dt, keywords=kw,
                            query_configs=qc)
        afi.connect()

        item = afi.get_items().__next__()

        processed = afi.process_item(item)

        afi.update_targets([processed])

        dt_data = list(dt.query())

        self.assertEqual(len(dt_data), 1)

    def test_default_query_configs(self):
        from propnet.dbtools.aflow_ingester_defaults import default_query_configs
        from aflow import K
        for config_ in default_query_configs:
            # Check that no error occurs when making query
            AflowIngester._get_query_obj(config_['catalog'],
                                         config_['k'],
                                         config_['exclude'],
                                         config_['filter'])
            for kw in config_['select']:
                self.assertIsNotNone(getattr(K, kw))


if __name__ == '__main__':
    unittest.main()
