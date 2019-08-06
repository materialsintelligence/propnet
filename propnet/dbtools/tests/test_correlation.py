import unittest
import os

import json
from monty.json import jsanitize
from monty.serialization import loadfn
from maggma.stores import MemoryStore
from maggma.runner import Runner

from itertools import product

from propnet.models import add_builtin_models_to_registry
from propnet.dbtools.correlation import CorrelationBuilder

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'test_data')
PROPNET_PROPS = ["band_gap_pbe", "atomic_density", "bulk_modulus", "vickers_hardness"]

@unittest.skip
class CorrelationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        add_builtin_models_to_registry()
        cls.propnet_props = PROPNET_PROPS

        cls.propstore = MemoryStore()
        cls.propstore.connect()
        materials_file = os.path.join(TEST_DATA_DIR, "correlation_propnet_data.json")
        if os.path.exists(materials_file):
            with open(materials_file, 'r') as f:
                materials = json.load(f)
            materials = jsanitize(materials, strict=True, allow_bson=True)
            cls.propstore.update(materials)

        cls.quantity_store = MemoryStore()
        cls.quantity_store.connect()
        quantities_file = os.path.join(TEST_DATA_DIR, "correlation_propnet_quantity_data.json")
        if os.path.exists(quantities_file):
            with open(quantities_file, 'r') as f:
                quantities = json.load(f)
            quantities = jsanitize(quantities, strict=True, allow_bson=True)
            cls.quantity_store.update(quantities, key='internal_id')

        cls.correlation = None

        # vickers hardness (x-axis) vs. bulk modulus (y-axis)
        cls.correlation_values_vickers_bulk = {
            'linlsq': 0.49302857395714916,
            'pearson': 0.7021599347421844,
            'spearman': 0.7658801470036752,
            'mic': 0.7069707677575771,
            'theilsen': 0.4498696692135892,
            'ransac': 0.412850153689996
        }

        cls.correlation_values_bulk_vickers = {
            'linlsq': 0.49302857395714916,
            'pearson': 0.7021599347421844,
            'spearman': 0.7658801470036752,
            'mic': 0.7069707677575771,
            'theilsen': 0.46461142885951223,
            'ransac': 0.3699261228882036
        }

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(os.path.join(TEST_DATA_DIR, "test_output.json")):
            os.remove(os.path.join(TEST_DATA_DIR, "test_output.json"))

    def setUp(self):
        self.correlation = MemoryStore()
        self.correlation.connect()

    def test_serial_runner(self):
        builder = CorrelationBuilder(self.propstore, self.correlation,
                                     from_quantity_db=False)
        runner = Runner([builder])
        runner.run()

    def test_serial_runner_quantity_db(self):
        # This only runs over the 4 properties in the database because
        # the mongomock db cannot be indexed and is therefore very slow
        builder = CorrelationBuilder(self.quantity_store, self.correlation,
                                     props=self.propnet_props,
                                     from_quantity_db=True)
        runner = Runner([builder])
        runner.run()

    def test_multiproc_runner(self):
        builder = CorrelationBuilder(self.propstore, self.correlation,
                                     from_quantity_db=False)
        runner = Runner([builder], max_workers=4)
        runner.run()

    def test_multiproc_runner_quantity_db(self):
        # This only runs over the 4 properties in the database because
        # the mongomock db cannot be indexed and is therefore very slow
        builder = CorrelationBuilder(self.quantity_store, self.correlation,
                                     props=self.propnet_props,
                                     from_quantity_db=True)
        runner = Runner([builder], max_workers=4)
        runner.run()

    def test_process_item(self):
        test_props = [['band_gap_pbe', 'atomic_density'],
                      ['bulk_modulus', 'vickers_hardness']]
        linlsq_correlation_values = [0.011294932700383722, 0.49302857395714916]
        path_lengths = [None, 2]

        for source_db, is_quantity_db in zip((self.propstore, self.quantity_store),
                                             (False, True)):
            for props, expected_correlation_val, expected_path_length in \
                    zip(test_props, linlsq_correlation_values, path_lengths):
                builder = CorrelationBuilder(source_db, self.correlation,
                                             props=props,
                                             from_quantity_db=is_quantity_db)
                processed = None
                prop_x, prop_y = props
                for item in builder.get_items():
                    if item['x_name'] == prop_x and \
                            item['y_name'] == prop_y:
                        processed = builder.process_item(item)
                        break
                # print(processed)
                self.assertIsNotNone(processed)
                self.assertIsInstance(processed, tuple)
                px, py, correlation, func_name, n_points, path_length = processed
                self.assertEqual(px, prop_x)
                self.assertEqual(py, prop_y)
                self.assertAlmostEqual(correlation, expected_correlation_val)
                self.assertEqual(func_name, 'linlsq')
                self.assertEqual(n_points, 200)
                self.assertEqual(path_length, expected_path_length)

    def test_correlation_funcs(self):
        def custom_correlation_func(x, y):
            return 0.5

        correlation_values = {k: v for k, v in self.correlation_values_bulk_vickers.items()}
        correlation_values['test_correlation.custom_correlation_func'] = 0.5

        builder = CorrelationBuilder(self.propstore, self.correlation,
                                     props=['vickers_hardness', 'bulk_modulus'],
                                     funcs=['all', custom_correlation_func],
                                     from_quantity_db=False)

        self.assertEqual(set(builder._funcs.keys()), set(correlation_values.keys()),
                         msg="Are there new built-in functions in the correlation builder?")

        for item in builder.get_items():
            if item['x_name'] == 'bulk_modulus' and \
                    item['y_name'] == 'vickers_hardness':
                processed = builder.process_item(item)
                self.assertIsInstance(processed, tuple)
                prop_x, prop_y, correlation, func_name, n_points, path_length = processed
                self.assertEqual(prop_x, 'bulk_modulus')
                self.assertEqual(prop_y, 'vickers_hardness')
                self.assertIn(func_name, correlation_values.keys())
                self.assertAlmostEqual(correlation, correlation_values[func_name])
                self.assertEqual(n_points, 200)
                self.assertEqual(path_length, 2)

    def test_database_and_file_write(self):
        builder = CorrelationBuilder(self.propstore, self.correlation,
                                     props=self.propnet_props,
                                     funcs='all',
                                     out_file=os.path.join(TEST_DATA_DIR, "test_output.json"),
                                     from_quantity_db=False)

        runner = Runner([builder])
        runner.run()

        # Test database output
        data = list(self.correlation.query(criteria={}))
        # count = n_props**2 * n_funcs
        # n_props = 4, n_funcs = 6
        self.assertEqual(len(data), 96, msg="Are there new built-in funcs in the builder?")

        for d in data:
            self.assertIsInstance(d, dict)
            self.assertEqual(set(d.keys()), {'property_x', 'property_y',
                                             'correlation', 'correlation_func',
                                             'n_points', 'shortest_path_length',
                                             'id', '_id', 'last_updated'})
            self.assertEqual(d['n_points'], 200)
            if d['property_x'] == 'vickers_hardness' and \
                    d['property_y'] == 'bulk_modulus':
                # print("{}: {}".format(d['correlation_func'], d['correlation']))
                self.assertAlmostEqual(
                    d['correlation'],
                    self.correlation_values_vickers_bulk[d['correlation_func']])
            elif d['property_x'] == 'bulk_modulus' and \
                    d['property_y'] == 'vickers_hardness':
                # print("{}: {}".format(d['correlation_func'], d['correlation']))
                self.assertAlmostEqual(
                    d['correlation'],
                    self.correlation_values_bulk_vickers[d['correlation_func']])
        # Test file output
        expected_file_data = loadfn(os.path.join(TEST_DATA_DIR, 'correlation_outfile.json'))
        actual_file_data = loadfn(os.path.join(TEST_DATA_DIR, 'test_output.json'))

        self.assertIsInstance(actual_file_data, dict)
        self.assertEqual(actual_file_data.keys(), expected_file_data.keys())
        self.assertEqual(set(actual_file_data['properties']), set(expected_file_data['properties']))

        expected_props = expected_file_data['properties']
        actual_props = actual_file_data['properties']

        for prop_x, prop_y in product(expected_props, repeat=2):
            iex, iey = expected_props.index(prop_x), expected_props.index(prop_y)
            iax, iay = actual_props.index(prop_x), actual_props.index(prop_y)

            self.assertEqual(actual_file_data['n_points'][iax][iay],
                             expected_file_data['n_points'][iex][iey])
            self.assertEqual(actual_file_data['shortest_path_length'][iax][iay],
                             expected_file_data['shortest_path_length'][iex][iey])

            for f in builder._funcs.keys():
                self.assertAlmostEqual(actual_file_data['correlation'][f][iax][iay],
                                       expected_file_data['correlation'][f][iex][iey])

    def test_sample_size_limit(self):
        sample_sizes = [50, 300]
        expected_n_points = [50, 200]

        for sample_size, n_points in zip(sample_sizes, expected_n_points):
            correlation_store = MemoryStore()
            builder = CorrelationBuilder(self.propstore, correlation_store,
                                         props=['bulk_modulus', 'vickers_hardness'],
                                         funcs='linlsq', sample_size=sample_size,
                                         from_quantity_db=False)
            runner = Runner([builder])
            runner.run()

            data = list(correlation_store.query(criteria={}))
            for d in data:
                self.assertEqual(d['n_points'], n_points)

        with self.assertRaises(ValueError):
            _ = CorrelationBuilder(self.propstore, self.correlation, sample_size=1)


if __name__ == "__main__":
    unittest.main()
