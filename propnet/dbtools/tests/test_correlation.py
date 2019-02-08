import unittest
import os

from monty.serialization import loadfn
from monty.json import jsanitize
from maggma.stores import MemoryStore
from maggma.runner import Runner

from propnet.dbtools.correlation import CorrelationBuilder

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


class CorrelationTest(unittest.TestCase):
    def setUp(self):
        self.propstore = MemoryStore()
        self.propstore.connect()
        materials = loadfn(os.path.join(TEST_DIR, "correlation_propnet_data.json"))
        materials = jsanitize(materials, strict=True, allow_bson=True)
        self.propstore.update(materials)
        self.materials = MemoryStore()
        self.materials.connect()
        materials = loadfn(os.path.join(TEST_DIR, "correlation_mp_data.json"))
        materials = jsanitize(materials, strict=True, allow_bson=True)
        self.materials.update(materials)
        self.correlation = MemoryStore()
        self.correlation.connect()
        self.propnet_props = ["band_gap_pbe", "bulk_modulus", "vickers_hardness"]
        self.mp_query_props = ["magnetism.total_magnetization_normalized_vol"]
        self.mp_props = ["total_magnetization_normalized_vol"]

        # vickers hardness (x-axis) vs. bulk modulus (y-axis)
        self.correlation_values_vickers_bulk = {
            'linlsq': 0.4155837083845686,
            'pearson': 0.6446578227126143,
            'spearman': 0.6975924398109954,
            'mic': 0.5616515521782413,
            'theilsen': 0.4047519736540858,
            'ransac': 0.3747245847179631}

        self.correlation_values_bulk_vickers = {
            'linlsq': 0.4155837083845686,
            'pearson': 0.6446578227126143,
            'spearman': 0.6975924398109954,
            'mic': 0.5616515521782413,
            'theilsen': 0.39860109570815505,
            'ransac': 0.3119656700613579}

    def test_serial_runner(self):
        builder = CorrelationBuilder(self.propstore, self.materials, self.correlation)
        runner = Runner([builder])
        runner.run()

    def test_multiproc_runner(self):
        builder = CorrelationBuilder(self.propstore, self.materials, self.correlation)
        runner = Runner([builder], max_workers=2)
        runner.run()

    def test_process_item(self):
        test_props = [['band_gap_pbe', 'total_magnetization_normalized_vol'],
                      ['bulk_modulus', 'vickers_hardness']]
        linlsq_correlation_values = [0.03620401274778131, 0.4155837083845686]
        path_lengths = [None, 2]

        for props, expected_correlation_val, expected_path_length in \
                zip(test_props, linlsq_correlation_values, path_lengths):
            builder = CorrelationBuilder(self.propstore, self.materials, self.correlation,
                                         props=props)
            processed = None
            prop_x, prop_y = props
            for item in builder.get_items():
                if item['x_name'] == prop_x and \
                        item['y_name'] == prop_y:
                    processed = builder.process_item(item)
                    break

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

        builder = CorrelationBuilder(self.propstore, self.materials, self.correlation,
                                     props=['vickers_hardness', 'bulk_modulus'],
                                     funcs=['all', custom_correlation_func])

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

    def test_database_write(self):
        builder = CorrelationBuilder(self.propstore, self.materials, self.correlation,
                                     props=self.propnet_props + self.mp_props,
                                     funcs='all')

        runner = Runner([builder])
        runner.run()

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
                self.assertAlmostEqual(
                    d['correlation'],
                    self.correlation_values_vickers_bulk[d['correlation_func']])
            elif d['property_x'] == 'bulk_modulus' and \
                    d['property_y'] == 'vickers_hardness':
                self.assertAlmostEqual(
                    d['correlation'],
                    self.correlation_values_bulk_vickers[d['correlation_func']])

    # Just here for reference, in case anyone wants to create a new set
    # of test materials. Requires mongogrant read access to knowhere.lbl.gov.
    @unittest.skipIf(True, "Skipping test materials creation")
    def create_test_docs(self):
        from maggma.advanced_stores import MongograntStore
        from monty.serialization import dumpfn
        pnstore = MongograntStore("ro:knowhere.lbl.gov/mp_core", "propnet")
        pnstore.connect()
        mpstore = MongograntStore("ro:knowhere.lbl.gov/mp_core", "materials")
        mpstore.connect()
        cursor = pnstore.query(
            criteria={'$and': [
                {'$or': [{p: {'$exists': True}},
                         {'inputs.symbol_type': p}]}
                for p in self.propnet_props]},
            properties=['task_id'])
        pn_mpids = [item['task_id'] for item in cursor]
        cursor = mpstore.query(criteria={p: {'$exists': True} for p in self.mp_query_props},
                               properties=['task_id'])
        mp_mpids = [item['task_id'] for item in cursor]
        mpids = list(set(pn_mpids).intersection(set(mp_mpids)))[:200]
        pn_data = pnstore.query(criteria={'task_id': {'$in': mpids}},
                                properties=['task_id', 'inputs'] +
                                           [p + '.mean' for p in self.propnet_props] +
                                           [p + '.units' for p in self.propnet_props])
        dumpfn(list(pn_data), os.path.join(TEST_DIR, "correlation_propnet_data.json"))
        mp_data = mpstore.query(criteria={'task_id': {'$in': mpids}},
                                properties=['task_id'] + self.mp_query_props)
        dumpfn(list(mp_data), os.path.join(TEST_DIR, "correlation_mp_data.json"))
