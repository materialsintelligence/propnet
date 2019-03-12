import unittest
import os

import json
from monty.json import jsanitize
from monty.serialization import loadfn
from maggma.stores import MemoryStore
from maggma.runner import Runner

from itertools import product

from propnet.dbtools.correlation import CorrelationBuilder

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


class CorrelationTest(unittest.TestCase):

    def setUp(self):
        self.propnet_props = ["band_gap_pbe", "bulk_modulus", "vickers_hardness"]
        self.mp_query_props = ["magnetism.total_magnetization_normalized_vol"]
        self.mp_props = ["total_magnetization_normalized_vol"]

        self.propstore = MemoryStore()
        self.propstore.connect()
        with open(os.path.join(TEST_DIR, "correlation_propnet_data.json"), 'r') as f:
            materials = json.load(f)
        materials = jsanitize(materials, strict=True, allow_bson=True)
        self.propstore.update(materials)
        self.materials = MemoryStore()
        self.materials.connect()
        with open(os.path.join(TEST_DIR, "correlation_mp_data.json"), 'r') as f:
            materials = json.load(f)
        materials = jsanitize(materials, strict=True, allow_bson=True)
        self.materials.update(materials)
        self.correlation = MemoryStore()
        self.correlation.connect()

        # vickers hardness (x-axis) vs. bulk modulus (y-axis)
        self.correlation_values_vickers_bulk = {
            'linlsq': 0.07030669243379202,
            'pearson': 0.2651540918669593,
            'spearman': 0.6759408985224631,
            'mic': 0.5529971905082182,
            'theilsen': -3.9351770244782456,
            'ransac': -4.528702228127463}

        self.correlation_values_bulk_vickers = {
            'linlsq': 0.07030669243379202,
            'pearson': 0.2651540918669593,
            'spearman': 0.6759408985224631,
            'mic': 0.5529971905082182,
            'theilsen': 0.040612225849504746,
            'ransac': 0.04576997520687298}

    def tearDown(self):
        if os.path.exists(os.path.join(TEST_DIR, "test_output.json")):
            os.remove(os.path.join(TEST_DIR, "test_output.json"))

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
        linlsq_correlation_values = [0.03522007675120975, 0.07030669243379202]
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

    def test_database_and_file_write(self):
        builder = CorrelationBuilder(self.propstore, self.materials, self.correlation,
                                     props=self.propnet_props + self.mp_props,
                                     funcs='all',
                                     out_file=os.path.join(TEST_DIR, "test_output.json"))

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
        expected_file_data = loadfn(os.path.join(TEST_DIR, 'correlation_outfile.json'))
        actual_file_data = loadfn(os.path.join(TEST_DIR, 'test_output.json'))

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
                                           [p + '.units' for p in self.propnet_props] +
                                           [p + '.quantities' for p in self.propnet_props])
        dumpfn(list(pn_data), os.path.join(TEST_DIR, "correlation_propnet_data.json"))
        mp_data = mpstore.query(criteria={'task_id': {'$in': mpids}},
                                properties=['task_id'] + self.mp_query_props)
        dumpfn(list(mp_data), os.path.join(TEST_DIR, "correlation_mp_data.json"))
