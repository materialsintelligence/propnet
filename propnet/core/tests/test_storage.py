import unittest
import os

import numpy as np

from propnet.core.storage import StorageQuantity, ProvenanceStore, ProvenanceStoreQuantity
from propnet.core.symbols import Symbol
from propnet.core.quantity import QuantityFactory, NumQuantity
from propnet.core.provenance import ProvenanceElement
from propnet import ureg

from monty.json import jsanitize, MontyDecoder


TEST_DIR = os.path.dirname(os.path.abspath(__file__))


class StorageTest(unittest.TestCase):
    def setUp(self):
        # Most of this setup is verbatim from the GraphTest class
        symbols = StorageTest.generate_canonical_symbols()

        a = [QuantityFactory.create_quantity(symbols['A'], 19),
             QuantityFactory.create_quantity(symbols['A'], 23)]
        b = [QuantityFactory.create_quantity(symbols['B'], 38,
                                             provenance=ProvenanceElement(model='model1',
                                                                          inputs=[a[0]])),
             QuantityFactory.create_quantity(symbols['B'], 46,
                                             provenance=ProvenanceElement(model='model1',
                                                                          inputs=[a[1]]))]
        c = [QuantityFactory.create_quantity(symbols['C'], 57,
                                             provenance=ProvenanceElement(model='model1',
                                                                          inputs=[a[0]])),
             QuantityFactory.create_quantity(symbols['C'], 69,
                                             provenance=ProvenanceElement(model='model1',
                                                                          inputs=[a[1]]))]
        g = [QuantityFactory.create_quantity(symbols['G'], 95,
                                             provenance=ProvenanceElement(model='model2',
                                                                          inputs=[a[0]])),
             QuantityFactory.create_quantity(symbols['G'], 115,
                                             provenance=ProvenanceElement(model='model2',
                                                                          inputs=[a[1]]))]
        f = [QuantityFactory.create_quantity(symbols['F'], 266,
                                             provenance=ProvenanceElement(model='model3',
                                                                          inputs=[b[0]])),
             QuantityFactory.create_quantity(symbols['F'], 322,
                                             provenance=ProvenanceElement(model='model3',
                                                                          inputs=[b[1]]))]
        d = [QuantityFactory.create_quantity(symbols['D'], 23826,
                                             provenance=ProvenanceElement(model='model4',
                                                                          inputs=[b[0], c[0]])),
             QuantityFactory.create_quantity(symbols['D'], 28842,
                                             provenance=ProvenanceElement(model='model4',
                                                                          inputs=[b[0], c[1]])),
             QuantityFactory.create_quantity(symbols['D'], 28842,
                                             provenance=ProvenanceElement(model='model4',
                                                                          inputs=[b[1], c[0]])),
             QuantityFactory.create_quantity(symbols['D'], 34914,
                                             provenance=ProvenanceElement(model='model4',
                                                                          inputs=[b[1], c[1]])),
             QuantityFactory.create_quantity(symbols['D'], 70395,
                                             provenance=ProvenanceElement(model='model5',
                                                                          inputs=[c[0], g[0]])),
             QuantityFactory.create_quantity(symbols['D'], 85215,
                                             provenance=ProvenanceElement(model='model5',
                                                                          inputs=[c[0], g[1]])),
             QuantityFactory.create_quantity(symbols['D'], 85215,
                                             provenance=ProvenanceElement(model='model5',
                                                                          inputs=[c[1], g[0]])),
             QuantityFactory.create_quantity(symbols['D'], 103155,
                                             provenance=ProvenanceElement(model='model5',
                                                                          inputs=[c[1], g[1]]))]
        self.expected_quantities = a + b + c + d + f + g
        self.quantity_with_uncertainty = NumQuantity.from_weighted_mean(d)
        obj_symbol = Symbol("B", category='object')
        self.object_quantity = QuantityFactory.create_quantity(obj_symbol, "Test string")
        self.maxDiff = None

    @staticmethod
    def generate_canonical_symbols():
        """
        Returns a set of Symbol objects used in testing.
        Returns: (dict<str, Symbol>)
        """
        A = Symbol('A', ['A'], ['A'], units="dimensionless", shape=[1])
        B = Symbol('B', ['B'], ['B'], units="dimensionless", shape=[1])
        C = Symbol('C', ['C'], ['C'], units="dimensionless", shape=[1])
        D = Symbol('D', ['D'], ['D'], units="dimensionless", shape=[1])
        G = Symbol('G', ['G'], ['G'], units="dimensionless", shape=[1])
        F = Symbol('F', ['F'], ['F'], units="dimensionless", shape=[1])
        return {
            'A': A,
            'B': B,
            'C': C,
            'D': D,
            'G': G,
            'F': F
        }

    def test_storage_quantity_from_quantity(self):
        quantities = self.expected_quantities
        d = quantities[10]

        storage_quantity = StorageQuantity.from_quantity(d)

        self.assertIsInstance(storage_quantity, StorageQuantity)
        self.assertEqual(storage_quantity._data_type, "NumQuantity")
        # This checks value equality
        self.assertEqual(storage_quantity.symbol, d.symbol)
        self.assertTrue(np.isclose(storage_quantity.value, d.value))
        self.assertListEqual(storage_quantity.tags, d.tags)
        # This checks __eq__() and that __eq__() commutes
        self.assertEqual(storage_quantity, d)
        self.assertEqual(d, storage_quantity)
        # This checks types and values explicitly in provenance to make sure everything was built correctly.
        # It is more robust than __eq__()
        self.rec_provenance_tree_check(storage_quantity.provenance, d.provenance)

        q = self.quantity_with_uncertainty
        storage_quantity_with_uncertainty = StorageQuantity.from_quantity(q)

        self.assertIsInstance(storage_quantity_with_uncertainty, StorageQuantity)
        self.assertEqual(storage_quantity_with_uncertainty._data_type, "NumQuantity")
        self.assertEqual(storage_quantity_with_uncertainty.symbol, q.symbol)
        self.assertTrue(np.isclose(storage_quantity_with_uncertainty.value, q.value))
        self.assertListEqual(storage_quantity_with_uncertainty.tags, q.tags)
        self.assertIsNotNone(storage_quantity_with_uncertainty.uncertainty)
        self.assertIsInstance(storage_quantity_with_uncertainty.uncertainty, ureg.Quantity)
        self.assertEqual(storage_quantity_with_uncertainty, q)

        # Test ObjQuantity coercion
        q = self.object_quantity
        storage_quantity_object = StorageQuantity.from_quantity(q)

        self.assertIsInstance(storage_quantity_object, StorageQuantity)
        self.assertEqual(storage_quantity_object._data_type, "ObjQuantity")
        self.assertEqual(storage_quantity_object.value, q.value)
        self.assertEqual(storage_quantity_object.symbol, q.symbol)
        self.assertListEqual(storage_quantity_object.tags, q.tags)
        self.assertIsNone(storage_quantity_object.units)
        self.assertEqual(storage_quantity_object, q)

    def test_from_provenance_element(self):
        quantities = self.expected_quantities
        # a = quantities[0]
        # b = quantities[2]
        # c = quantities[4]
        d = quantities[10]

        storage_provenance = ProvenanceStore.from_provenance_element(d.provenance)
        # This checks __eq__() for provenance, and that __eq__() commutes
        self.assertEqual(storage_provenance, d.provenance)
        self.assertEqual(d.provenance, storage_provenance)

        self.rec_provenance_tree_check(storage_provenance, d.provenance)

    def test_provenance_storage_quantity_from_quantity(self):
        quantities = self.expected_quantities
        d = quantities[10]

        storage_quantity = ProvenanceStoreQuantity.from_quantity(d)

        self.assertIsInstance(storage_quantity, ProvenanceStoreQuantity)
        self.assertEqual(storage_quantity._data_type, "NumQuantity")
        # This checks value equality
        self.assertEqual(storage_quantity.symbol, d.symbol)
        self.assertTrue(np.isclose(storage_quantity.value, d.value))
        self.assertListEqual(storage_quantity.tags, d.tags)
        self.assertFalse(storage_quantity.is_from_dict())
        self.assertTrue(storage_quantity.is_value_retrieved())
        # This checks __eq__() and that __eq__() commutes
        self.assertEqual(storage_quantity, d)
        self.assertEqual(d, storage_quantity)
        # This checks types and values explicitly in provenance to make sure everything was built correctly.
        # It is more robust than __eq__()
        self.rec_provenance_tree_check(storage_quantity.provenance, d.provenance)

        q = self.quantity_with_uncertainty
        storage_quantity_with_uncertainty = ProvenanceStoreQuantity.from_quantity(q)

        self.assertIsInstance(storage_quantity_with_uncertainty, StorageQuantity)
        self.assertEqual(storage_quantity_with_uncertainty._data_type, "NumQuantity")
        self.assertEqual(storage_quantity_with_uncertainty.symbol, q.symbol)
        self.assertTrue(np.isclose(storage_quantity_with_uncertainty.value, q.value))
        self.assertListEqual(storage_quantity_with_uncertainty.tags, q.tags)
        self.assertFalse(storage_quantity.is_from_dict())
        self.assertTrue(storage_quantity.is_value_retrieved())
        self.assertIsNotNone(storage_quantity_with_uncertainty.uncertainty)
        self.assertIsInstance(storage_quantity_with_uncertainty.uncertainty, ureg.Quantity)
        self.assertEqual(storage_quantity_with_uncertainty, q)

        # Test ObjQuantity coercion
        q = self.object_quantity
        storage_quantity_object = ProvenanceStoreQuantity.from_quantity(q)

        self.assertIsInstance(storage_quantity_object, StorageQuantity)
        self.assertEqual(storage_quantity_object._data_type, "ObjQuantity")
        self.assertEqual(storage_quantity_object.value, q.value)
        self.assertEqual(storage_quantity_object.symbol, q.symbol)
        self.assertListEqual(storage_quantity_object.tags, q.tags)
        self.assertFalse(storage_quantity.is_from_dict())
        self.assertTrue(storage_quantity.is_value_retrieved())
        self.assertIsNone(storage_quantity_object.units)
        self.assertEqual(storage_quantity_object, q)

    def test_as_dict_from_dict_from_json(self):
        # Test with non-canonical symbol
        quantities = self.expected_quantities
        original_quantity = quantities[2]
        q = StorageQuantity.from_quantity(original_quantity)
        d = q.as_dict()
        self.assertIsInstance(d['symbol_type'], Symbol)
        self.assertDictEqual(d, {"@module": "propnet.core.storage",
                                 "@class": "StorageQuantity",
                                 "internal_id": q._internal_id,
                                 "data_type": "NumQuantity",
                                 "symbol_type": q.symbol,
                                 "value": 38,
                                 "units": "dimensionless",
                                 "provenance": q.provenance,
                                 "tags": [],
                                 "uncertainty": None})

        q_from_dict = StorageQuantity.from_dict(d)
        self.assertIsInstance(q_from_dict, StorageQuantity)
        self.assertEqual(q_from_dict._data_type, "NumQuantity")
        self.assertEqual(q_from_dict.symbol, q.symbol)
        self.assertTrue(np.isclose(q_from_dict.value, q.value))
        self.assertListEqual(q_from_dict.tags, q.tags)
        self.assertEqual(q_from_dict, q)
        self.rec_provenance_tree_check(q_from_dict.provenance, original_quantity.provenance)

        json_dict = jsanitize(q, strict=True)
        self.assertDictEqual(
            json_dict, {"@module": "propnet.core.storage",
                        "@class": "StorageQuantity",
                        "internal_id": q._internal_id,
                        "data_type": "NumQuantity",
                        "symbol_type": {"@module": "propnet.core.symbols",
                                        "@class": "Symbol",
                                        "name": "B",
                                        "display_names": ["B"],
                                        "display_symbols": ["B"],
                                        "units": [1, []],
                                        "shape": [1],
                                        "object_type": None,
                                        "comment": None,
                                        "category": "property",
                                        "constraint": None,
                                        "default_value": None},
                        "value": 38,
                        "units": "dimensionless",
                        "provenance": {"@module": "propnet.core.storage",
                                       "@class": "ProvenanceStore",
                                       "model": "model1",
                                       "source": q.provenance.source,
                                       "inputs": [{'@module': 'propnet.core.storage',
                                                    '@class': 'ProvenanceStoreQuantity',
                                                    'data_type': 'NumQuantity',
                                                    'symbol_type': {'@module': 'propnet.core.symbols',
                                                                    '@class': 'Symbol',
                                                                    'name': 'A',
                                                                    'display_names': ['A'],
                                                                    'display_symbols': ['A'],
                                                                    'units': [1, []],
                                                                    'shape': [1],
                                                                    'object_type': None,
                                                                    'comment': None,
                                                                    'category': 'property',
                                                                    'constraint': None,
                                                                    'default_value': None},
                                                    'internal_id': q.provenance.inputs[0]._internal_id,
                                                    'tags': [],
                                                    'provenance': {'@module': 'propnet.core.storage',
                                                                   '@class': 'ProvenanceStore',
                                                                   'model': None,
                                                                   'inputs': None,
                                                                   'source': q.provenance.inputs[
                                                                       0].provenance.source}}]},
                        "tags": [],
                        "uncertainty": None})

        q_from_json_dict = MontyDecoder().process_decoded(json_dict)
        self.assertIsInstance(q_from_json_dict, StorageQuantity)
        self.assertEqual(q_from_json_dict._data_type, "NumQuantity")
        self.assertEqual(q_from_json_dict.symbol, q.symbol)
        self.assertTrue(np.isclose(q_from_json_dict.value, q.value))
        self.assertListEqual(q_from_json_dict.tags, q.tags)
        self.assertEqual(q_from_json_dict.provenance, original_quantity.provenance)
        self.assertEqual(q_from_json_dict, q)
        self.rec_provenance_tree_check(q_from_json_dict.provenance, original_quantity.provenance,
                                       check_from_dict=True)

    def test_value_lookup(self):
        pass

    def test_default_instantiation(self):
        pass

    def rec_provenance_tree_check(self, q_storage, q_original, check_from_dict=False):
        self.assertIsInstance(q_storage, ProvenanceStore)
        self.assertEqual(q_storage.model, q_original.model)
        for v in q_storage.inputs or []:
            self.assertIsInstance(v, ProvenanceStoreQuantity)
            v_orig = [x for x in q_original.inputs
                      if x._internal_id == v._internal_id]
            self.assertEqual(len(v_orig), 1)
            v_orig = v_orig[0]
            if check_from_dict:
                self.assertTrue(v.is_from_dict())
                self.assertFalse(v.is_value_retrieved())
            else:
                self.assertTrue(np.isclose(v.value, v_orig.value))
            self.assertListEqual(v.tags, v_orig.tags)
            self.rec_provenance_tree_check(v.provenance, v_orig.provenance, check_from_dict)
