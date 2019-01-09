import unittest
import os

import numpy as np

from propnet.core.storage import StorageQuantity, ProvenanceStore, ProvenanceStoreQuantity
from propnet.core.symbols import Symbol
from propnet.core.quantity import QuantityFactory, NumQuantity
from propnet.core.provenance import ProvenanceElement
from propnet.models import DEFAULT_MODEL_DICT
from propnet import ureg

from monty.json import jsanitize, MontyDecoder
import copy


TEST_DIR = os.path.dirname(os.path.abspath(__file__))


class StorageTest(unittest.TestCase):
    def setUp(self):
        # Most of this setup is verbatim from the GraphTest class
        symbols = StorageTest.generate_symbols()

        self.custom_syms_as_dicts = {
            k: {'@module': 'propnet.core.symbols',
                '@class': 'Symbol',
                'name': k,
                'display_names': [k],
                'display_symbols': [k],
                'units': (1, ()),
                'shape': [1],
                'object_type': None,
                'comment': None,
                'category': 'property',
                'constraint': None,
                'default_value': None} for k in ['A', 'B']
        }

        self.custom_symbols_json = self.custom_syms_as_dicts
        for k in ['A', 'B']:
            self.custom_symbols_json[k]['units'] = [1, []]

        a = [QuantityFactory.create_quantity(symbols['A'], 19),
             QuantityFactory.create_quantity(symbols['A'], 23)]
        b = [QuantityFactory.create_quantity(symbols['B'], 38,
                                             provenance=ProvenanceElement(model='model1',
                                                                          inputs=[a[0]])),
             QuantityFactory.create_quantity(symbols['B'], 46,
                                             provenance=ProvenanceElement(model='model1',
                                                                          inputs=[a[1]]))]
        self.quantities_custom_symbol = {"A": a,
                                         "B": b}

        self.sq_custom_sym_as_dicts = {
            k: [{'@module': 'propnet.core.storage',
                 '@class': 'StorageQuantity',
                 'internal_id': vv._internal_id,
                 'data_type': 'NumQuantity',
                 'symbol_type': symbols[k],
                 'value': vv.magnitude,
                 'units': 'dimensionless',
                 'provenance': ProvenanceStore.from_provenance_element(vv.provenance),
                 'tags': [],
                 'uncertainty': None} for vv in v] for k, v in self.quantities_custom_symbol.items()
        }

        provenances_json = {
            "A": [{'@module': 'propnet.core.storage',
                   '@class': 'ProvenanceStore',
                   'model': None,
                   'inputs': None,
                   'source': aa.provenance.source} for aa in a]}
        provenances_json['B'] = [
            {'@module': 'propnet.core.storage',
             '@class': 'ProvenanceStore',
             'model': 'model1',
             'inputs': [{'@module': 'propnet.core.storage',
                         '@class': 'ProvenanceStoreQuantity',
                         'data_type': 'NumQuantity',
                         'symbol_type': self.custom_symbols_json['A'],
                         'internal_id': q.provenance.inputs[0]._internal_id,
                         'tags': [],
                         'provenance': p}],
             'source': q.provenance.source} for q, p in zip(b, provenances_json['A'])]

        self.sq_custom_sym_json = copy.deepcopy(self.sq_custom_sym_as_dicts)
        for sym in ['A', 'B']:
            for q, p in zip(self.sq_custom_sym_json[sym], provenances_json[sym]):
                q['symbol_type'] = self.custom_symbols_json[sym]
                q['provenance'] = p

        band_gaps = [QuantityFactory.create_quantity('band_gap', 3.3, 'eV'),
                     QuantityFactory.create_quantity('band_gap', 2.1, 'eV')]

        bg_ri_model = DEFAULT_MODEL_DICT['band_gap_refractive_index_moss']
        refractive_indices = [bg_ri_model.evaluate({"Eg": bg}).pop('refractive_index') for bg in band_gaps]

        self.quantities_canonical_symbol = {"band_gaps": band_gaps,
                                            "refractive_indices": refractive_indices}

        self.sq_canonical_sym_as_dicts_no_value = copy.deepcopy(self.sq_custom_sym_as_dicts)
        self.sq_canonical_sym_as_dicts_no_value['band_gaps'] = self.sq_canonical_sym_as_dicts_no_value.pop('A')
        self.sq_canonical_sym_as_dicts_no_value['refractive_indices'] = self.sq_canonical_sym_as_dicts_no_value.pop('B')

        for d, sq in zip(self.sq_canonical_sym_as_dicts_no_value['band_gaps'], band_gaps):
            d.update({
                "internal_id": sq._internal_id,
                "symbol_type": "band_gap",
                "units": "electron_volt",
                "provenance": ProvenanceStore.from_provenance_element(sq.provenance)
            })
            d.pop('value')

        for d, sq in zip(self.sq_canonical_sym_as_dicts_no_value['refractive_indices'], refractive_indices):
            d.update({
                "internal_id": sq._internal_id,
                "symbol_type": "refractive_index",
                "units": "dimensionless",
                "provenance": ProvenanceStore.from_provenance_element(sq.provenance)
            })
            d.pop('value')

        self.sq_canonical_sym_values = {"band_gaps": [3.3, 2.1],
                                        "refractive_indices": [2.316340583741216, 2.593439239956374]}

        provenances_json = {
            "band_gaps": [{'@module': 'propnet.core.storage',
                           '@class': 'ProvenanceStore',
                           'model': None,
                           'inputs': None,
                           'source': bg.provenance.source}
                          for bg in band_gaps]
        }
        provenances_json['refractive_indices'] = [{
            '@module': 'propnet.core.storage',
            '@class': 'ProvenanceStore',
            'model': 'band_gap_refractive_index_moss',
            'inputs': [{'@module': 'propnet.core.storage',
                        '@class': 'ProvenanceStoreQuantity',
                        'data_type': 'NumQuantity',
                        'symbol_type': 'band_gap',
                        'internal_id': bg._internal_id,
                        'tags': [],
                        'provenance': pj}],
            'source': ri.provenance.source} for bg, pj, ri in zip(band_gaps,
                                                                  provenances_json['band_gaps'],
                                                                  refractive_indices)
        ]

        self.sq_canonical_sym_json_no_value = copy.deepcopy(self.sq_canonical_sym_as_dicts_no_value)

        for sym in ["band_gaps", "refractive_indices"]:
            for q, p in zip(self.sq_canonical_sym_json_no_value[sym], provenances_json[sym]):
                q['provenance'] = p

        self.quantity_with_uncertainty = NumQuantity.from_weighted_mean(b)
        obj_symbol = symbols['C']
        self.object_quantity = QuantityFactory.create_quantity(obj_symbol, "Test string")

        # This setting allows dict differences to be shown in full
        self.maxDiff = None

    @staticmethod
    def generate_symbols():
        """
        Returns a set of Symbol objects used in testing.
        Returns: (dict<str, Symbol>)
        """
        a = Symbol('A', ['A'], ['A'], units="dimensionless", shape=[1])
        b = Symbol('B', ['B'], ['B'], units="dimensionless", shape=[1])
        c = Symbol('C', ['C'], ['C'], category="object", object_type=str)

        return {
            'A': a,
            'B': b,
            'C': c
        }

    def test_internal_dicts(self):
        for sym, q in self.quantities_custom_symbol.items():
            for i in range(2):
                compare_dict = StorageQuantity.from_quantity(q[i]).as_dict()
                self.assertDictEqual(compare_dict, self.sq_custom_sym_as_dicts[sym][i])
                compare_dict = jsanitize(StorageQuantity.from_quantity(q[i]), strict=True)
                self.assertDictEqual(compare_dict, self.sq_custom_sym_json[sym][i])

        for sym, q in self.quantities_canonical_symbol.items():
            for i in range(2):
                compare_dict = {k: v for k, v in
                                StorageQuantity.from_quantity(q[i]).as_dict().items()
                                if k != "value"}
                self.assertDictEqual(compare_dict, self.sq_canonical_sym_as_dicts_no_value[sym][i])
                compare_dict = {k: v for k, v in
                                jsanitize(StorageQuantity.from_quantity(q[i]), strict=True).items()
                                if k != "value"}
                self.assertDictEqual(compare_dict, self.sq_canonical_sym_json_no_value[sym][i])

    @unittest.skip
    def test_storage_quantity_from_quantity(self):
        quantities = self.quantities_custom_symbol
        q = quantities[10]

        storage_quantity = StorageQuantity.from_quantity(q)

        self.assertIsInstance(storage_quantity, StorageQuantity)
        self.assertEqual(storage_quantity._data_type, "NumQuantity")
        # This checks value equality
        self.assertEqual(storage_quantity.symbol, q.symbol)
        self.assertIsInstance(storage_quantity.value, int)
        self.assertTrue(storage_quantity.value, q.magnitude)
        self.assertListEqual(storage_quantity.tags, q.tags)
        # This checks __eq__() and that __eq__() commutes
        self.assertEqual(storage_quantity, q)
        self.assertEqual(q, storage_quantity)
        # This checks types and values explicitly in provenance to make sure everything was built correctly.
        # It is more robust than __eq__()
        self.rec_provenance_tree_check(storage_quantity.provenance, q.provenance)

        q = self.quantity_with_uncertainty
        storage_quantity_with_uncertainty = StorageQuantity.from_quantity(q)

        self.assertIsInstance(storage_quantity_with_uncertainty, StorageQuantity)
        self.assertEqual(storage_quantity_with_uncertainty._data_type, "NumQuantity")
        self.assertEqual(storage_quantity_with_uncertainty.symbol, q.symbol)
        self.assertIsInstance(storage_quantity_with_uncertainty.value, float)
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
        self.assertIsInstance(storage_quantity_object.value, str)
        self.assertEqual(storage_quantity_object.value, q.value)
        self.assertEqual(storage_quantity_object.symbol, q.symbol)
        self.assertListEqual(storage_quantity_object.tags, q.tags)
        self.assertIsNone(storage_quantity_object.units)
        self.assertEqual(storage_quantity_object, q)

    @unittest.skip
    def test_from_provenance_element(self):
        quantities = self.quantities_custom_symbol
        # a = quantities[0]
        # b = quantities[2]
        # c = quantities[4]
        q = quantities[10]

        storage_provenance = ProvenanceStore.from_provenance_element(q.provenance)
        # This checks __eq__() for provenance, and that __eq__() commutes
        self.assertEqual(storage_provenance, q.provenance)
        self.assertEqual(q.provenance, storage_provenance)

        self.rec_provenance_tree_check(storage_provenance, q.provenance)

    @unittest.skip
    def test_provenance_storage_quantity_from_quantity(self):
        quantities = self.quantities_custom_symbol
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

    @unittest.skip
    def test_as_dict_from_dict_from_json(self):
        # Test with non-canonical symbol
        quantities = self.quantities_custom_symbol
        original_quantity = quantities[2]
        q = StorageQuantity.from_quantity(original_quantity)
        d = q.as_dict()
        expected_dict = {"@module": "propnet.core.storage",
                         "@class": "StorageQuantity",
                         "internal_id": q._internal_id,
                         "data_type": "NumQuantity",
                         "symbol_type": q.symbol,
                         "value": 38,
                         "units": "dimensionless",
                         "provenance": q.provenance,
                         "tags": [],
                         "uncertainty": None}
        self.assertIsInstance(d['symbol_type'], Symbol)
        self.assertDictEqual(d, expected_dict)

        q_from_dict = StorageQuantity.from_dict(d)
        self.assertIsInstance(q_from_dict, StorageQuantity)
        self.assertEqual(q_from_dict._data_type, "NumQuantity")
        self.assertEqual(q_from_dict.symbol, q.symbol)
        self.assertTrue(np.isclose(q_from_dict.value, q.magnitude))
        self.assertListEqual(q_from_dict.tags, q.tags)
        self.assertEqual(q_from_dict, q)
        self.rec_provenance_tree_check(q_from_dict.provenance, original_quantity.provenance)

        json_dict = jsanitize(q, strict=True)
        expected_json_dict = {"@module": "propnet.core.storage",
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
                              "uncertainty": None}
        self.assertDictEqual(json_dict, expected_json_dict)

        q_from_json_dict = MontyDecoder().process_decoded(json_dict)
        self.assertIsInstance(q_from_json_dict, StorageQuantity)
        self.assertEqual(q_from_json_dict._data_type, "NumQuantity")
        self.assertEqual(q_from_json_dict.symbol, q.symbol)
        self.assertTrue(np.isclose(q_from_json_dict.value, q.magnitude))
        self.assertListEqual(q_from_json_dict.tags, q.tags)
        self.assertEqual(q_from_json_dict.provenance, original_quantity.provenance)
        self.assertEqual(q_from_json_dict, q)
        self.rec_provenance_tree_check(q_from_json_dict.provenance, original_quantity.provenance,
                                       check_from_dict=True)

        # Test with canonical symbol
        original_quantity = self.quantities_canonical_symbol[2]
        q = StorageQuantity.from_quantity(original_quantity)
        d = q.as_dict()
        expected_dict = {"@module": "propnet.core.storage",
                         "@class": "StorageQuantity",
                         "internal_id": q._internal_id,
                         "data_type": "NumQuantity",
                         "symbol_type": "refractive_index",
                         "value": 1.3,
                         "units": "dimensionless",
                         "provenance": q.provenance,
                         "tags": [],
                         "uncertainty": None}

        self.assertIsInstance(d['symbol_type'], str)
        self.assertDictEqual(d, expected_dict)

        q_from_dict = StorageQuantity.from_dict(d)
        self.assertIsInstance(q_from_dict, StorageQuantity)
        self.assertEqual(q_from_dict._data_type, "NumQuantity")
        self.assertEqual(q_from_dict.symbol, q.symbol)
        self.assertTrue(np.isclose(q_from_dict.value, q.magnitude))
        self.assertListEqual(q_from_dict.tags, q.tags)
        self.assertEqual(q_from_dict, q)
        self.rec_provenance_tree_check(q_from_dict.provenance, original_quantity.provenance)

        json_dict = jsanitize(q, strict=True)

        canonical_expected_json_dict = expected_json_dict
        canonical_expected_json_dict.update({"symbol_type": "refractive_index",
                                             "value": 1.3,
                                             "units": "dimensionless",
                                             "internal_id": q._internal_id})
        canonical_expected_json_dict['provenance'].update({"model": "modelA",
                                                           "source": q.provenance.source})
        canonical_expected_json_dict['provenance']['inputs'][0].update({"symbol_type": "band_gap",
                                                                        "internal_id": q.provenance.inputs[
                                                                            0]._internal_id})
        canonical_expected_json_dict['provenance']['inputs'][0]['provenance']['source'] = q.provenance.inputs[
            0].provenance.source

        self.assertDictEqual(json_dict, canonical_expected_json_dict)

        q_from_json_dict = MontyDecoder().process_decoded(json_dict)
        self.assertIsInstance(q_from_json_dict, StorageQuantity)
        self.assertEqual(q_from_json_dict._data_type, "NumQuantity")
        self.assertEqual(q_from_json_dict.symbol, q.symbol)
        self.assertTrue(np.isclose(q_from_json_dict.magnitude, q.magnitude))
        self.assertListEqual(q_from_json_dict.tags, q.tags)
        self.assertEqual(q_from_json_dict.provenance, original_quantity.provenance)
        self.assertEqual(q_from_json_dict, q)
        self.rec_provenance_tree_check(q_from_json_dict.provenance, original_quantity.provenance,
                                       check_from_dict=True)

        # Test with symbol with uncertainty
        original_quantity = self.quantity_with_uncertainty
        q = StorageQuantity.from_quantity(original_quantity)
        d = q.as_dict()
        expected_dict = {"@module": "propnet.core.storage",
                         "@class": "StorageQuantity",
                         "internal_id": q._internal_id,
                         "data_type": "NumQuantity",
                         "symbol_type": q.symbol,
                         "units": "dimensionless",
                         "provenance": q.provenance,
                         "tags": []}
        self.assertIsInstance(d['symbol_type'], Symbol)
        self.assertTrue(np.isclose(d['value'], 42.0))
        self.assertTrue(np.isclose(d['uncertainty'][0], 4.0))
        self.assertEqual(d['uncertainty'][1], ())
        value = d.pop('value')
        uncertainty = d.pop('uncertainty')
        self.assertDictEqual(d, expected_dict)
        d['value'] = value
        d['uncertainty'] = uncertainty

        q_from_dict = StorageQuantity.from_dict(d)
        self.assertIsInstance(q_from_dict, StorageQuantity)
        self.assertEqual(q_from_dict._data_type, "NumQuantity")
        self.assertEqual(q_from_dict.symbol, q.symbol)
        self.assertTrue(np.isclose(q_from_dict.value, q.magnitude))
        self.assertTrue(np.isclose(q_from_dict.uncertainty.magnitude,
                                   q.uncertainty.magnitude))
        np.isclose(q_from_dict.uncertainty, q.uncertainty)
        self.assertEqual(q_from_dict.uncertainty.units,
                         q.uncertainty.units)
        self.assertListEqual(q_from_dict.tags, q.tags)
        self.assertEqual(q_from_dict, q)
        self.rec_provenance_tree_check(q_from_dict.provenance, original_quantity.provenance)

        json_dict = jsanitize(q, strict=True)

        self.assertTrue(np.isclose(json_dict['value'], 42.0))
        self.assertTrue(np.isclose(json_dict['uncertainty'][0], 4.0))
        self.assertEqual(json_dict['uncertainty'][1], [])
        value = json_dict.pop('value')
        uncertainty = json_dict.pop('uncertainty')
        uncertainty_expected_json_dict = expected_json_dict
        uncertainty_expected_json_dict['internal_id'] = q._internal_id
        uncertainty_expected_json_dict['provenance'].update({"model": "aggregation",
                                                             "source": q.provenance.source})
        uncertainty_expected_json_dict['provenance']['inputs'] = [
            {'@module': 'propnet.core.storage',
             '@class': 'ProvenanceStoreQuantity',
             'data_type': 'NumQuantity',
             'symbol_type': uncertainty_expected_json_dict['symbol'],
             'internal_id': q.provenance.inputs[0]._internal_id,
             'tags': [],
             'provenance': {'@module': 'propnet.core.storage', '@class': 'ProvenanceStore', 'model': 'model1',
                            'inputs': [
                                {'@module': 'propnet.core.storage', '@class': 'ProvenanceStoreQuantity',
                                 'data_type': 'NumQuantity',
                                 'symbol_type': {'@module': 'propnet.core.symbols', '@class': 'Symbol', 'name': 'A',
                                                 'display_names': ['A'], 'display_symbols': ['A'], 'units': [1, []],
                                                 'shape': [1],
                                                 'object_type': None, 'comment': None, 'category': 'property',
                                                 'constraint': None,
                                                 'default_value': None},
                                 'internal_id': 'c4f2349738bf45e0bbcc4e169e5958aa', 'tags': [],
                                 'provenance': {'@module': 'propnet.core.storage', '@class': 'ProvenanceStore',
                                                'model': None,
                                                'inputs': None,
                                                'source': {'source': None,
                                                           'source_key': 'c4f2349738bf45e0bbcc4e169e5958aa',
                                                           'date_created': '2019-01-08 12:19:47'}}}],
                            'source': {'source': None, 'date_created': '2019-01-08 12:19:47',
                                       'source_key': '3ab8d320a80a4a709ef408e44ca28a2e'}}},
            {'@module': 'propnet.core.storage', '@class': 'ProvenanceStoreQuantity', 'data_type': 'NumQuantity',
             'symbol_type': {'@module': 'propnet.core.symbols', '@class': 'Symbol', 'name': 'B', 'display_names': ['B'],
                             'display_symbols': ['B'], 'units': [1, []], 'shape': [1], 'object_type': None,
                             'comment': None, 'category': 'property', 'constraint': None, 'default_value': None},
             'internal_id': 'dadb18e94336472497dcb12e9e00d616', 'tags': [],
             'provenance': {'@module': 'propnet.core.storage', '@class': 'ProvenanceStore', 'model': 'model1',
                            'inputs': [
                                {'@module': 'propnet.core.storage', '@class': 'ProvenanceStoreQuantity',
                                 'data_type': 'NumQuantity',
                                 'symbol_type': {'@module': 'propnet.core.symbols', '@class': 'Symbol', 'name': 'A',
                                                 'display_names': ['A'], 'display_symbols': ['A'], 'units': [1, []],
                                                 'shape': [1],
                                                 'object_type': None, 'comment': None, 'category': 'property',
                                                 'constraint': None,
                                                 'default_value': None},
                                 'internal_id': 'c894fcedb48e4eee87e26bb75b5d02f8', 'tags': [],
                                 'provenance': {'@module': 'propnet.core.storage', '@class': 'ProvenanceStore',
                                                'model': None,
                                                'inputs': None,
                                                'source': {'source': None,
                                                           'source_key': 'c894fcedb48e4eee87e26bb75b5d02f8',
                                                           'date_created': '2019-01-08 12:19:47'}}}],
                            'source': {'source': None, 'date_created': '2019-01-08 12:19:47',
                                       'source_key': 'dadb18e94336472497dcb12e9e00d616'}}}]

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
            elif NumQuantity.is_acceptable_type(v.value):
                self.assertTrue(np.isclose(v.value, v_orig.value))
            else:
                self.assertEqual(v.value, v_orig.value)
            self.assertListEqual(v.tags, v_orig.tags)
            self.rec_provenance_tree_check(v.provenance, v_orig.provenance, check_from_dict)
