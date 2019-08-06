import unittest
import os
import copy
from itertools import chain
import warnings

import numpy as np
from monty.json import jsanitize, MontyDecoder
from pint import UnitStrippedWarning

from propnet.dbtools.storage import StorageQuantity, ProvenanceStore, ProvenanceStoreQuantity
from propnet.core.symbols import Symbol
from propnet.core.quantity import QuantityFactory, NumQuantity, BaseQuantity
from propnet.core.provenance import ProvenanceElement
from propnet import ureg
# noinspection PyUnresolvedReferences
from propnet.models import add_builtin_models_to_registry
from propnet.core.registry import Registry

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


class StorageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Suppress warnings when using numpy functions with pint Quantities
        warnings.filterwarnings("ignore", category=UnitStrippedWarning)
        add_builtin_models_to_registry()
        # Inspiration was taken from the GraphTest class
        # I tried to construct the dictionaries for comparison
        # without writing out every one explicitly by reusing
        # information where it was applicable.
        # If this is too unreadable, can change to writing it
        # out explicitly in a JSON file and importing it. Would
        # still need to replace some fields dynamically.
        symbols = StorageTest.generate_symbols()

        cls.custom_syms_as_dicts = {
            k: {'@module': 'propnet.core.symbols',
                '@class': 'Symbol',
                'name': k,
                'display_names': [k],
                'display_symbols': [k],
                'units': (1, ()),
                'shape': 1,
                'object_type': None,
                'comment': None,
                'category': 'property',
                'constraint': None,
                'default_value': None,
                'is_builtin': False} for k in ['A', 'B', 'C']
        }
        cls.custom_syms_as_dicts['C'].update(
            {"units": None,
             "shape": None,
             "object_type": "str",
             "category": "object"})

        cls.custom_symbols_json = copy.deepcopy(cls.custom_syms_as_dicts)
        for k in ['A', 'B']:
            cls.custom_symbols_json[k]['units'] = [1, []]

        a = [QuantityFactory.create_quantity(symbols['A'], 19),
             QuantityFactory.create_quantity(symbols['A'], 23)]
        b = [QuantityFactory.create_quantity(symbols['B'], 38,
                                             provenance=ProvenanceElement(model='model1',
                                                                          inputs=[a[0]])),
             QuantityFactory.create_quantity(symbols['B'], 46,
                                             provenance=ProvenanceElement(model='model1',
                                                                          inputs=[a[1]]))]
        cls.quantities_custom_symbol = {"A": a,
                                         "B": b}

        cls.sq_custom_sym_as_dicts = {
            k: [{'@module': 'propnet.dbtools.storage',
                 '@class': 'StorageQuantity',
                 'internal_id': vv._internal_id,
                 'data_type': 'NumQuantity',
                 'symbol_type': symbols[k],
                 'value': vv.magnitude,
                 'units': 'dimensionless',
                 'provenance': ProvenanceStore.from_provenance_element(vv.provenance),
                 'tags': [],
                 'uncertainty': None} for vv in v] for k, v in cls.quantities_custom_symbol.items()
        }

        provenances_json = {
            "A": [{'@module': 'propnet.dbtools.storage',
                   '@class': 'ProvenanceStore',
                   'model': None,
                   'inputs': None,
                   'source': aa.provenance.source} for aa in a]}
        provenances_json['B'] = [
            {'@module': 'propnet.dbtools.storage',
             '@class': 'ProvenanceStore',
             'model': 'model1',
             'inputs': [{'@module': 'propnet.dbtools.storage',
                         '@class': 'ProvenanceStoreQuantity',
                         'data_type': 'NumQuantity',
                         'symbol_type': cls.custom_symbols_json['A'],
                         'internal_id': q.provenance.inputs[0]._internal_id,
                         'tags': [],
                         'provenance': p}],
             'source': q.provenance.source} for q, p in zip(b, provenances_json['A'])]

        cls.sq_custom_sym_json = copy.deepcopy(cls.sq_custom_sym_as_dicts)
        for sym in ['A', 'B']:
            for q, p in zip(cls.sq_custom_sym_json[sym], provenances_json[sym]):
                q['symbol_type'] = cls.custom_symbols_json[sym]
                q['provenance'] = p

        band_gaps = [QuantityFactory.create_quantity('band_gap', 3.3, 'eV'),
                     QuantityFactory.create_quantity('band_gap', 2.1, 'eV')]

        bg_ri_model = Registry("models")['band_gap_refractive_index_moss']
        refractive_indices = [bg_ri_model.evaluate({"Eg": bg}).pop('refractive_index') for bg in band_gaps]

        cls.quantities_canonical_symbol = {"band_gaps": band_gaps,
                                            "refractive_indices": refractive_indices}

        cls.sq_canonical_sym_as_dicts_no_value = copy.deepcopy(cls.sq_custom_sym_as_dicts)
        cls.sq_canonical_sym_as_dicts_no_value['band_gaps'] = cls.sq_canonical_sym_as_dicts_no_value.pop('A')
        cls.sq_canonical_sym_as_dicts_no_value['refractive_indices'] = cls.sq_canonical_sym_as_dicts_no_value.pop('B')

        for d, sq in zip(cls.sq_canonical_sym_as_dicts_no_value['band_gaps'], band_gaps):
            d.update({
                "internal_id": sq._internal_id,
                "symbol_type": "band_gap",
                "units": "electron_volt",
                "provenance": ProvenanceStore.from_provenance_element(sq.provenance)
            })
            d.pop('value')

        for d, sq in zip(cls.sq_canonical_sym_as_dicts_no_value['refractive_indices'], refractive_indices):
            d.update({
                "internal_id": sq._internal_id,
                "symbol_type": "refractive_index",
                "units": "dimensionless",
                "provenance": ProvenanceStore.from_provenance_element(sq.provenance)
            })
            d.pop('value')

        cls.sq_canonical_sym_values = {"band_gaps": [3.3, 2.1],
                                        "refractive_indices": [2.316340583741216, 2.593439239956374]}

        provenances_json['band_gaps'] = [
            {'@module': 'propnet.dbtools.storage',
             '@class': 'ProvenanceStore',
             'model': None,
             'inputs': None,
             'source': bg.provenance.source}
            for bg in band_gaps
        ]

        provenances_json['refractive_indices'] = [{
            '@module': 'propnet.dbtools.storage',
            '@class': 'ProvenanceStore',
            'model': 'band_gap_refractive_index_moss',
            'inputs': [{'@module': 'propnet.dbtools.storage',
                        '@class': 'ProvenanceStoreQuantity',
                        'data_type': 'NumQuantity',
                        'symbol_type': 'band_gap',
                        'internal_id': bg._internal_id,
                        'tags': [],
                        'provenance': pj}],
            'source': ri.provenance.source}
            for bg, pj, ri in zip(band_gaps,
                                  provenances_json['band_gaps'],
                                  refractive_indices)
        ]

        cls.sq_canonical_sym_json_no_value = copy.deepcopy(cls.sq_canonical_sym_as_dicts_no_value)

        for sym in ["band_gaps", "refractive_indices"]:
            for q, p in zip(cls.sq_canonical_sym_json_no_value[sym], provenances_json[sym]):
                q['provenance'] = p

        cls.quantity_with_uncertainty = NumQuantity.from_weighted_mean(b)
        cls.sq_with_uncertainty_as_dict_no_numbers = {
            '@module': 'propnet.dbtools.storage',
            '@class': 'StorageQuantity',
            'internal_id': cls.quantity_with_uncertainty._internal_id,
            'data_type': 'NumQuantity',
            'symbol_type': symbols['B'],
            'units': 'dimensionless',
            'provenance': ProvenanceStore.from_provenance_element(
                cls.quantity_with_uncertainty.provenance),
            'tags': []}

        provenances_json = {
            '@module': 'propnet.dbtools.storage',
            '@class': 'ProvenanceStore',
            'model': 'aggregation',
            'inputs': [
                {'@module': 'propnet.dbtools.storage',
                 '@class': 'ProvenanceStoreQuantity',
                 'data_type': 'NumQuantity',
                 'symbol_type': cls.custom_symbols_json['B'],
                 'internal_id': b['internal_id'],
                 'tags': [],
                 'provenance': b['provenance']}
                for b in cls.sq_custom_sym_json['B']],
            'source': cls.quantity_with_uncertainty.provenance.source
        }

        cls.sq_with_uncertainty_json_no_numbers = copy.deepcopy(cls.sq_with_uncertainty_as_dict_no_numbers)
        cls.sq_with_uncertainty_json_no_numbers.update({"symbol_type": cls.custom_symbols_json['B'],
                                                         "provenance": provenances_json})
        cls.sq_with_uncertainty_numbers = {"value": 42.0,
                                            "uncertainty": 4.0}

        obj_symbol = symbols['C']
        cls.object_quantity = QuantityFactory.create_quantity(obj_symbol, "Test string")
        cls.sq_object_as_dict = copy.deepcopy(cls.sq_custom_sym_as_dicts['A'][0])
        cls.sq_object_as_dict.update({
            "data_type": "ObjQuantity",
            "symbol_type": symbols['C'],
            "internal_id": cls.object_quantity._internal_id,
            "value": "Test string",
            "units": None,
            "provenance": ProvenanceStore.from_provenance_element(cls.object_quantity.provenance)
        })
        cls.sq_object_json = copy.deepcopy(cls.sq_object_as_dict)
        cls.sq_object_json.update(
            {"symbol_type": cls.custom_syms_as_dicts['C'],
             "provenance": {'@module': 'propnet.dbtools.storage',
                            '@class': 'ProvenanceStore',
                            'model': None,
                            'inputs': None,
                            'source': cls.object_quantity.provenance.source}}
        )

        # This setting allows dict differences to be shown in full
        cls.maxDiff = None

    @classmethod
    def tearDownClass(cls):
        warnings.resetwarnings()
        non_builtin_syms = [k for k, v in Registry("symbols").items() if not v.is_builtin]
        for sym in non_builtin_syms:
            Registry("symbols").pop(sym)

    @staticmethod
    def generate_symbols():
        """
        Returns a set of Symbol objects used in testing.
        Returns: (dict<str, Symbol>)
        """
        a = Symbol('A', ['A'], ['A'], units="dimensionless", shape=[1])
        b = Symbol('B', ['B'], ['B'], units="dimensionless", shape=[1])
        c = Symbol('C', ['C'], ['C'], category="object", object_type=str)

        syms = {
            'A': a,
            'B': b,
            'C': c
        }

        for sym in syms.values():
            Registry("symbols")[sym] = sym

        return syms

    def test_default_instantiation(self):
        provenance_store = ProvenanceStore()
        self.assertIsNone(provenance_store.inputs)
        self.assertIsNone(provenance_store.model)
        self.assertIsNone(provenance_store.source)

        provenance_store_quantity = ProvenanceStoreQuantity()
        self.assertIsNone(provenance_store_quantity._data_type)
        self.assertIsNone(provenance_store_quantity.value)
        self.assertIsNone(provenance_store_quantity._internal_id)
        self.assertIsNone(provenance_store_quantity.provenance)
        self.assertIsNone(provenance_store_quantity.symbol)
        self.assertIsNone(provenance_store_quantity.tags)
        self.assertIsNone(provenance_store_quantity.uncertainty)
        self.assertIsNone(provenance_store_quantity.units)
        self.assertFalse(provenance_store_quantity.has_value())

        storage_quantity = StorageQuantity()
        self.assertIsNone(storage_quantity._data_type)
        self.assertIsNone(storage_quantity.value)
        self.assertIsNone(storage_quantity._internal_id)
        self.assertIsNone(storage_quantity.provenance)
        self.assertIsNone(storage_quantity.symbol)
        self.assertIsNone(storage_quantity.tags)
        self.assertIsNone(storage_quantity.uncertainty)
        self.assertIsNone(storage_quantity.units)

    def test_provenance_store_from_provenance_element(self):
        for q in chain.from_iterable(self.quantities_custom_symbol.values()):
            # Check both ways to instantiate
            storage_provenances = [ProvenanceStore.from_provenance_element(q.provenance),
                                   ProvenanceStore(q.provenance)]

            for storage_provenance in storage_provenances:
                # This checks __eq__() for provenance, and that __eq__() commutes
                self.assertEqual(storage_provenance, q.provenance)
                self.assertEqual(q.provenance, storage_provenance)
                self.rec_provenance_tree_check(storage_provenance, q.provenance)

            # Test copy of StorageQuantity with from_quantity
            p_in = ProvenanceStore(q.provenance)
            p_out = ProvenanceStore.from_provenance_element(p_in)

            self.assertIsNot(p_in, p_out)
            self.assertEqual(p_in, p_out)

        # Test fail with incorrect object type
        with self.assertRaises(TypeError):
            ProvenanceStore('Incorrect type')
        with self.assertRaises(TypeError):
            ProvenanceStore.from_provenance_element('Incorrect type')

    def test_provenance_storage_quantity_from_quantity(self):
        for q in chain.from_iterable(self.quantities_custom_symbol.values()):
            storage_quantities = [ProvenanceStoreQuantity.from_quantity(q),
                                  ProvenanceStoreQuantity(q)]
            for storage_quantity in storage_quantities:
                self.assertIsInstance(storage_quantity, ProvenanceStoreQuantity)
                self.assertEqual(storage_quantity._data_type, "NumQuantity")
                # This checks value equality
                self.assertEqual(storage_quantity.symbol, q.symbol)
                self.assertTrue(np.isclose(storage_quantity.value, q.value))
                self.assertListEqual(storage_quantity.tags, q.tags)
                self.assertTrue(storage_quantity.has_value())
                # This checks __eq__() and that __eq__() commutes
                self.assertEqual(storage_quantity, q)
                self.assertEqual(q, storage_quantity)
                # This checks types and values explicitly in provenance to make sure everything was built correctly.
                # It is more robust than __eq__()
                self.rec_provenance_tree_check(storage_quantity.provenance, q.provenance)

        q = self.quantity_with_uncertainty
        storage_quantities_with_uncertainty = [ProvenanceStoreQuantity.from_quantity(q),
                                               ProvenanceStoreQuantity(q)]
        for storage_quantity_with_uncertainty in storage_quantities_with_uncertainty:
            self.assertIsInstance(storage_quantity_with_uncertainty, StorageQuantity)
            self.assertEqual(storage_quantity_with_uncertainty._data_type, "NumQuantity")
            self.assertEqual(storage_quantity_with_uncertainty.symbol, q.symbol)
            self.assertTrue(np.isclose(storage_quantity_with_uncertainty.value, q.value))
            self.assertListEqual(storage_quantity_with_uncertainty.tags, q.tags)
            self.assertTrue(storage_quantity_with_uncertainty.has_value())
            self.assertIsNotNone(storage_quantity_with_uncertainty.uncertainty)
            self.assertIsInstance(storage_quantity_with_uncertainty.uncertainty, ureg.Quantity)
            self.assertEqual(storage_quantity_with_uncertainty, q)

        # Test ObjQuantity coercion
        q = self.object_quantity
        storage_quantities_object = [ProvenanceStoreQuantity.from_quantity(q),
                                     ProvenanceStoreQuantity(q)]
        for storage_quantity_object in storage_quantities_object:
            self.assertIsInstance(storage_quantity_object, StorageQuantity)
            self.assertEqual(storage_quantity_object._data_type, "ObjQuantity")
            self.assertEqual(storage_quantity_object.value, q.value)
            self.assertEqual(storage_quantity_object.symbol, q.symbol)
            self.assertListEqual(storage_quantity_object.tags, q.tags)
            self.assertTrue(storage_quantity_object.has_value())
            self.assertIsNone(storage_quantity_object.units)
            self.assertEqual(storage_quantity_object, q)

        # Test copy of StorageQuantity with from_quantity
        q_in = ProvenanceStoreQuantity(q)
        q_out = ProvenanceStoreQuantity.from_quantity(q_in)

        self.assertIsNot(q_in, q_out)
        self.assertEqual(q_in, q_out)

        # Test fail with incorrect object type
        with self.assertRaises(TypeError):
            ProvenanceStoreQuantity('Incorrect type')
        with self.assertRaises(TypeError):
            ProvenanceStoreQuantity.from_quantity('Incorrect type')

    def test_storage_quantity_from_quantity(self):
        for q in chain.from_iterable(self.quantities_custom_symbol.values()):
            storage_quantities = [StorageQuantity.from_quantity(q),
                                  StorageQuantity(q)]
            for storage_quantity in storage_quantities:
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
        storage_quantities_with_uncertainty = [StorageQuantity.from_quantity(q),
                                               StorageQuantity(q)]

        for storage_quantity_with_uncertainty in storage_quantities_with_uncertainty:
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
        storage_quantities_object = [StorageQuantity.from_quantity(q),
                                     StorageQuantity(q)]

        for storage_quantity_object in storage_quantities_object:
            self.assertIsInstance(storage_quantity_object, StorageQuantity)
            self.assertEqual(storage_quantity_object._data_type, "ObjQuantity")
            self.assertIsInstance(storage_quantity_object.value, str)
            self.assertEqual(storage_quantity_object.value, q.value)
            self.assertEqual(storage_quantity_object.symbol, q.symbol)
            self.assertListEqual(storage_quantity_object.tags, q.tags)
            self.assertIsNone(storage_quantity_object.units)
            self.assertEqual(storage_quantity_object, q)

        # Test copy of StorageQuantity with from_quantity
        q_in = StorageQuantity(q)
        q_out = StorageQuantity.from_quantity(q_in)

        self.assertIsNot(q_in, q_out)
        self.assertEqual(q_in, q_out)

        # Test fail with incorrect object type
        with self.assertRaises(TypeError):
            StorageQuantity('Incorrect type')
        with self.assertRaises(TypeError):
            StorageQuantity.from_quantity('Incorrect type')

    def test_as_dict_as_json(self):
        # Check symbols to be sure they are still ok
        for name, sym in self.generate_symbols().items():
            compare_dict = sym.as_dict()
            self.assertDictEqual(compare_dict, self.custom_syms_as_dicts[name])
            compare_dict = jsanitize(sym, strict=True)
            self.assertDictEqual(compare_dict, self.custom_symbols_json[name])

        # Quantities with custom symbols
        for sym, q in self.quantities_custom_symbol.items():
            for qq, expected_dict, expected_json in zip(q,
                                                        self.sq_custom_sym_as_dicts[sym],
                                                        self.sq_custom_sym_json[sym]):
                sq = StorageQuantity.from_quantity(qq)
                compare_dict = sq.as_dict()
                self.assertDictEqual(compare_dict, expected_dict)
                compare_dict = jsanitize(sq, strict=True)
                self.assertDictEqual(compare_dict, expected_json)

        # Quantities with canonical symbols directly calculated from a real model
        for sym, q in self.quantities_canonical_symbol.items():
            for qq, expected_dict, expected_json in zip(q,
                                                        self.sq_canonical_sym_as_dicts_no_value[sym],
                                                        self.sq_canonical_sym_json_no_value[sym]):
                sq = StorageQuantity.from_quantity(qq)
                compare_dict = sq.as_dict()
                self.assertTrue(np.isclose(qq.magnitude, compare_dict['value']))
                compare_dict.pop('value')
                self.assertDictEqual(compare_dict, expected_dict)
                compare_dict = jsanitize(sq, strict=True)
                self.assertTrue(np.isclose(qq.magnitude, compare_dict['value']))
                compare_dict.pop('value')
                self.assertDictEqual(compare_dict, expected_json)

        # Quantity with uncertainty (calculated from mean), using custom symbols
        sq = StorageQuantity.from_quantity(self.quantity_with_uncertainty)
        compare_dict = sq.as_dict()
        self.assertTrue(np.isclose(self.quantity_with_uncertainty.magnitude, compare_dict['value']))
        uncertainty_value = compare_dict['uncertainty']
        self.assertTrue(np.isclose(self.quantity_with_uncertainty.uncertainty.magnitude,
                                   uncertainty_value))
        compare_dict.pop('value')
        compare_dict.pop('uncertainty')
        self.assertDictEqual(self.sq_with_uncertainty_as_dict_no_numbers, compare_dict)

        compare_dict = jsanitize(sq, strict=True)
        self.assertTrue(np.isclose(self.quantity_with_uncertainty.magnitude, compare_dict['value']))
        uncertainty_value = compare_dict['uncertainty']
        self.assertTrue(np.isclose(self.quantity_with_uncertainty.uncertainty.magnitude,
                                   uncertainty_value))
        compare_dict.pop('value')
        compare_dict.pop('uncertainty')
        self.assertDictEqual(self.sq_with_uncertainty_json_no_numbers, compare_dict)

        # Quantity that is an object, using a custom symbol
        sq = StorageQuantity.from_quantity(self.object_quantity)
        compare_dict = sq.as_dict()
        self.assertDictEqual(self.sq_object_as_dict, compare_dict)
        compare_dict = jsanitize(sq, strict=True)
        self.assertDictEqual(self.sq_object_json, compare_dict)

    def test_from_dict_from_json(self):
        # Test with non-canonical symbol
        for original_quantity in chain.from_iterable(self.quantities_custom_symbol.values()):
            q = StorageQuantity.from_quantity(original_quantity)
            d = q.as_dict()
            q_from_dict = StorageQuantity.from_dict(d)
            self.assertIsInstance(q_from_dict, StorageQuantity)
            self.assertEqual(q_from_dict._data_type, "NumQuantity")
            self.assertEqual(q_from_dict.symbol, q.symbol)
            self.assertTrue(np.isclose(q_from_dict.value, q.magnitude))
            self.assertEqual(q_from_dict.units, q.units)
            self.assertListEqual(q_from_dict.tags, q.tags)
            self.assertEqual(q_from_dict, q)
            self.rec_provenance_tree_check(q_from_dict.provenance, original_quantity.provenance)

            json_dict = jsanitize(q, strict=True)
            q_from_json_dict = MontyDecoder().process_decoded(json_dict)
            self.assertIsInstance(q_from_json_dict, StorageQuantity)
            self.assertEqual(q_from_json_dict._data_type, "NumQuantity")
            self.assertEqual(q_from_json_dict.symbol, q.symbol)

            self.assertTrue(np.isclose(q_from_json_dict.value, q.magnitude))
            self.assertEqual(q_from_json_dict.units, q.units)
            self.assertListEqual(q_from_json_dict.tags, q.tags)
            self.assertEqual(q_from_json_dict.provenance, original_quantity.provenance)
            self.assertEqual(q_from_json_dict, q)
            self.rec_provenance_tree_check(q_from_json_dict.provenance, original_quantity.provenance,
                                           from_dict=True)

        # Test with canonical symbol
        for original_quantity in chain.from_iterable(self.quantities_canonical_symbol.values()):
            q = StorageQuantity.from_quantity(original_quantity)
            d = q.as_dict()
            q_from_dict = StorageQuantity.from_dict(d)
            self.assertIsInstance(q_from_dict, StorageQuantity)
            self.assertEqual(q_from_dict._data_type, "NumQuantity")
            self.assertEqual(q_from_dict.symbol, q.symbol)
            self.assertTrue(np.isclose(q_from_dict.value, q.magnitude))
            self.assertEqual(q_from_dict.units, q.units)
            self.assertListEqual(q_from_dict.tags, q.tags)
            self.assertEqual(q_from_dict, q)
            self.rec_provenance_tree_check(q_from_dict.provenance, original_quantity.provenance)

            json_dict = jsanitize(q, strict=True)
            q_from_json_dict = MontyDecoder().process_decoded(json_dict)
            self.assertIsInstance(q_from_json_dict, StorageQuantity)
            self.assertEqual(q_from_json_dict._data_type, "NumQuantity")
            self.assertEqual(q_from_json_dict.symbol, q.symbol)
            self.assertTrue(np.isclose(q_from_json_dict.magnitude, q.magnitude))
            self.assertEqual(q_from_json_dict.units, q.units)
            self.assertListEqual(q_from_json_dict.tags, q.tags)
            self.assertEqual(q_from_json_dict.provenance, original_quantity.provenance)
            self.assertEqual(q_from_json_dict, q)
            self.rec_provenance_tree_check(q_from_json_dict.provenance, original_quantity.provenance,
                                           from_dict=True)

        # Test with quantity with uncertainty, custom symbol
        original_quantity = self.quantity_with_uncertainty
        q = StorageQuantity.from_quantity(original_quantity)
        d = q.as_dict()
        q_from_dict = StorageQuantity.from_dict(d)
        self.assertIsInstance(q_from_dict, StorageQuantity)
        self.assertEqual(q_from_dict._data_type, "NumQuantity")
        self.assertEqual(q_from_dict.symbol, q.symbol)
        self.assertTrue(np.isclose(q_from_dict.value, q.magnitude))
        self.assertEqual(q_from_dict.units, q.units)
        self.assertTrue(np.isclose(q_from_dict.uncertainty, q.uncertainty))
        self.assertListEqual(q_from_dict.tags, q.tags)
        self.assertEqual(q_from_dict, q)
        self.rec_provenance_tree_check(q_from_dict.provenance, original_quantity.provenance)

        json_dict = jsanitize(q, strict=True)
        q_from_json_dict = MontyDecoder().process_decoded(json_dict)
        self.assertIsInstance(q_from_json_dict, StorageQuantity)
        self.assertEqual(q_from_json_dict._data_type, "NumQuantity")
        self.assertEqual(q_from_json_dict.symbol, q.symbol)
        self.assertTrue(np.isclose(q_from_json_dict.magnitude, q.magnitude))
        self.assertEqual(q_from_json_dict.units, q.units)
        self.assertTrue(np.isclose(q_from_json_dict.uncertainty, q.uncertainty))
        self.assertListEqual(q_from_json_dict.tags, q.tags)
        self.assertEqual(q_from_json_dict.provenance, original_quantity.provenance)
        self.assertEqual(q_from_json_dict, q)
        self.rec_provenance_tree_check(q_from_json_dict.provenance, original_quantity.provenance,
                                       from_dict=True)

        # Test with object quantity
        original_quantity = self.object_quantity
        q = StorageQuantity.from_quantity(original_quantity)
        d = q.as_dict()
        q_from_dict = StorageQuantity.from_dict(d)
        self.assertIsInstance(q_from_dict, StorageQuantity)
        self.assertEqual(q_from_dict._data_type, "ObjQuantity")
        self.assertEqual(q_from_dict.symbol, q.symbol)
        self.assertEqual(q_from_dict.value, q.value)
        self.assertListEqual(q_from_dict.tags, q.tags)
        self.assertEqual(q_from_dict, q)
        self.rec_provenance_tree_check(q_from_dict.provenance, original_quantity.provenance)

        json_dict = jsanitize(q, strict=True)
        q_from_json_dict = MontyDecoder().process_decoded(json_dict)
        self.assertIsInstance(q_from_json_dict, StorageQuantity)
        self.assertEqual(q_from_json_dict._data_type, "ObjQuantity")
        self.assertEqual(q_from_json_dict.symbol, q.symbol)
        self.assertEqual(q_from_json_dict.value, q.value)
        self.assertListEqual(q_from_json_dict.tags, q.tags)
        self.assertEqual(q_from_json_dict, q)
        self.rec_provenance_tree_check(q_from_json_dict.provenance, original_quantity.provenance,
                                       from_dict=True)

    def test_value_lookup_to_quantity(self):
        def rec_verify_lookup(p_lookup, p_original):
            self.assertIsInstance(p_lookup, ProvenanceElement)
            for v in p_lookup.inputs or []:
                self.assertIsInstance(v, BaseQuantity)
                v_orig = [x for x in p_original.inputs
                          if x._internal_id == v._internal_id]
                self.assertEqual(len(v_orig), 1)
                v_orig = v_orig[0]
                self.assertIsNotNone(v.value)
                if isinstance(v, NumQuantity):
                    self.assertTrue(np.isclose(v.value, v_orig.value))
                    if v_orig.uncertainty:
                        self.assertTrue(np.isclose(v.uncertainty, v_orig.uncertainty))
                else:
                    self.assertEqual(v.value, v_orig.value)
                rec_verify_lookup(v.provenance, v_orig.provenance)

        lookup_dict = self.get_lookup_dict()
        lookup_fun = self.lookup_fun

        quantities = \
            list(chain.from_iterable(self.quantities_custom_symbol.values())) + \
            list(chain.from_iterable(self.quantities_canonical_symbol.values())) + \
            [self.quantity_with_uncertainty, self.object_quantity]

        for q in quantities:
            json_dict = jsanitize(StorageQuantity.from_quantity(q), strict=True)
            sq_json = MontyDecoder().process_decoded(json_dict)
            if sq_json.provenance.inputs:
                for v in sq_json.provenance.inputs:
                    self.assertIsNone(v.value)
            q_json_dict = sq_json.to_quantity(lookup=lookup_dict)
            q_json_fun = sq_json.to_quantity(lookup=lookup_fun)
            q_json_reconstruct_dict = StorageQuantity.reconstruct_quantity(json_dict, lookup_dict)
            q_json_reconstruct_fun = StorageQuantity.reconstruct_quantity(json_dict, lookup_fun)
            for q_json in (q_json_dict, q_json_fun, q_json_reconstruct_dict, q_json_reconstruct_fun):
                self.assertIsInstance(q_json, type(q))
                if isinstance(q_json, NumQuantity):
                    self.assertTrue(np.isclose(q_json.value, q.value))
                    if q.uncertainty:
                        self.assertTrue(np.isclose(q_json.uncertainty, q.uncertainty))
                else:
                    self.assertEqual(q_json.value, q.value)
                rec_verify_lookup(q_json.provenance, q.provenance)

            if q.provenance.inputs:
                with self.assertRaises(ValueError):
                    # Needs lookup but doesn't get a lookup container
                    sq_json.to_quantity()
                with self.assertRaises(ValueError):
                    # Needs lookup but doesn't get a lookup container
                    StorageQuantity.reconstruct_quantity(json_dict)

                with self.assertRaises(ValueError):
                    sq_json.to_quantity(lookup=self.lookup_fun_missing_value)

                with self.assertRaises(ValueError):
                    StorageQuantity.reconstruct_quantity(json_dict,
                                                         lookup=self.lookup_fun_missing_value)

                with self.assertRaises(TypeError):
                    sq_json.to_quantity(lookup=self.lookup_fun_incorrect_type)

                with self.assertRaises(TypeError):
                    StorageQuantity.reconstruct_quantity(json_dict,
                                                         lookup=self.lookup_fun_incorrect_type)

                key = q.provenance.inputs[0]._internal_id
                key_lookup = lookup_dict.pop(key)
                with self.assertRaises(ValueError):
                    sq_json.to_quantity(lookup=lookup_dict)
                with self.assertRaises(ValueError):
                    StorageQuantity.reconstruct_quantity(json_dict, lookup=lookup_dict)
                with self.assertRaises(ValueError):
                    sq_json.to_quantity(lookup=self.lookup_fun_key_not_found)
                with self.assertRaises(ValueError):
                    StorageQuantity.reconstruct_quantity(json_dict, lookup=self.lookup_fun_key_not_found)
                lookup_dict[key] = key_lookup

                with self.assertRaises(TypeError):
                    sq_json.to_quantity(lookup='This is not a lookup')
                with self.assertRaises(TypeError):
                    StorageQuantity.reconstruct_quantity(json_dict, lookup='This is not a lookup')

    def lookup_fun(self, key):
        return self.get_lookup_dict().get(key)

    def lookup_fun_missing_value(self, key):
        d = self.lookup_fun(key)
        if d:
            del d['value']
        return d

    @staticmethod
    def lookup_fun_incorrect_type(key):
        return key

    @staticmethod
    def lookup_fun_key_not_found(key):
        # Lookup function expects None when key is not found
        return None

    def get_lookup_dict(self):
        lookup_dict = {}
        quantities = \
            list(chain.from_iterable(self.quantities_custom_symbol.values())) + \
            list(chain.from_iterable(self.quantities_canonical_symbol.values())) + \
            [self.quantity_with_uncertainty, self.object_quantity]
        for q in quantities:
            lookup_dict[q._internal_id] = {"value": q.value,
                                           "units": q.units,
                                           "uncertainty": q.uncertainty}
        return lookup_dict

    def rec_provenance_tree_check(self, q_storage, q_original, from_dict=False):
        self.assertIsInstance(q_storage, ProvenanceStore)
        self.assertEqual(q_storage.model, q_original.model)
        for v in q_storage.inputs or []:
            self.assertIsInstance(v, ProvenanceStoreQuantity)
            v_orig = [x for x in q_original.inputs
                      if x._internal_id == v._internal_id]
            self.assertEqual(len(v_orig), 1)
            v_orig = v_orig[0]
            if from_dict:
                self.assertFalse(v.has_value())
            elif NumQuantity.is_acceptable_type(v.value):
                self.assertTrue(np.isclose(v.value, v_orig.value))
                self.assertTrue(v.has_value())
            else:
                self.assertEqual(v.value, v_orig.value)
                self.assertTrue(v.has_value())
            self.assertListEqual(v.tags, v_orig.tags)
            self.rec_provenance_tree_check(v.provenance, v_orig.provenance, from_dict)
