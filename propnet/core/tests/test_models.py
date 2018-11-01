import unittest

import math

from propnet.models import DEFAULT_MODEL_NAMES, DEFAULT_MODEL_DICT, DEFAULT_MODELS
from propnet.symbols import DEFAULT_SYMBOL_TYPE_NAMES
from propnet.core.models import EquationModel
from propnet.core.symbols import Symbol
from propnet.core.quantity import Quantity


# TODO: test PyModule, PyModel
# TODO: separate these into specific tests of model functionality
#       and validation of default models
class ModelTest(unittest.TestCase):
    def test_instantiate_all_models(self):
        models_to_test = []
        for model_name in DEFAULT_MODEL_NAMES:
            try:
                model = DEFAULT_MODEL_DICT.get(model_name)
                self.assertIsNotNone(model)
                models_to_test.append(model_name)
            except Exception as e:
                self.fail('Failed to load model {}: {}'.format(model_name, e))

    def test_model_formatting(self):
        # TODO: clean up tests (self.assertNotNone), test reference format too
        for model in DEFAULT_MODELS:
            self.assertIsNotNone(model.name)
            self.assertIsNotNone(model.categories)
            self.assertIsNotNone(model.description)
            self.assertIsNotNone(model.symbol_property_map)
            self.assertTrue(isinstance(model.symbol_property_map, dict))
            self.assertTrue(len(model.symbol_property_map.keys()) > 0)
            for key in model.symbol_property_map.keys():
                self.assertTrue(isinstance(key, str),
                                'Invalid symbol_property_map key: ' + str(key))
                self.assertTrue(isinstance(model.symbol_property_map[key], str) and
                                model.symbol_property_map[key] in DEFAULT_SYMBOL_TYPE_NAMES)
            self.assertTrue(model.connections is not None and isinstance(model.connections, list)
                            and len(model.connections) > 0)
            for reference in model.references:
                self.assertTrue(reference.startswith('@'))
            for item in model.connections:
                self.assertIsNotNone(item)
                self.assertTrue(isinstance(item, dict))
                self.assertTrue('inputs' in item.keys())
                self.assertTrue('outputs' in item.keys())
                self.assertIsNotNone(item['inputs'])
                self.assertIsNotNone(item['outputs'])
                self.assertTrue(isinstance(item['inputs'], list))
                self.assertTrue(isinstance(item['outputs'], list))
                self.assertTrue(len(item['inputs']) > 0)
                self.assertTrue(len(item['outputs']) > 0)
                for in_symb in item['inputs']:
                    self.assertIsNotNone(in_symb)
                    self.assertTrue(isinstance(in_symb, str))
                    self.assertTrue(in_symb in model.symbol_property_map.keys())
                for out_symb in item['outputs']:
                    self.assertIsNotNone(out_symb)
                    self.assertIsNotNone(isinstance(out_symb, str))
                    self.assertTrue(out_symb in model.symbol_property_map.keys())

    def test_validate_all_models(self):
        for model in DEFAULT_MODELS:
            self.assertTrue(model.validate_from_preset_test())

    def test_unit_handling(self):
        """
        Tests unit handling with a simple model that calculates the area of a rectangle as the
        product of two lengths.

        In this case the input lengths are provided in centimeters and meters.
        Tests whether the input units are properly coerced into canonical types.
        Tests whether the output units are properly set.
        Tests whether the model returns as predicted.
        Returns:
            None
        """
        L = Symbol('l', ['L'], ['L'], units=[1.0, [['centimeter', 1.0]]], shape=[1])
        A = Symbol('a', ['A'], ['A'], units=[1.0, [['centimeter', 2.0]]], shape=[1])
        get_area_config = {
            'name': 'area',
            # 'connections': [{'inputs': ['l1', 'l2'], 'outputs': ['a']}],
            'equations': ['a = l1 * l2'],
            # 'unit_map': {'l1': "cm", "l2": "cm", 'a': "cm^2"}
            'symbol_property_map': {"a": A, "l1": L, "l2": L}
        }
        model = EquationModel(**get_area_config)
        out = model.evaluate({'l1': Quantity(L, 1, 'meter'),
                              'l2': Quantity(L, 2)}, allow_failure=False)

        self.assertTrue(math.isclose(out['a'].magnitude, 200.0))
        self.assertTrue(out['a'].units == A.units)

    def test_example_code_helper(self):

        example_model = DEFAULT_MODEL_DICT['semi_empirical_mobility']

        # TODO: this is ugly, any way to fix it?
        example_code = """
from propnet.models import semi_empirical_mobility


K = 64
m_e = 0.009

semi_empirical_mobility.plug_in({
\t'K': K,
\t'm_e': m_e,
})
\"\"\"
returns {'mu_e': 8994.92312225673}
\"\"\"
"""
        self.assertEqual(example_model.example_code, example_code)

        for name, model in DEFAULT_MODEL_DICT.items():
            # A little weird, but otherwise you don't get the explicit error
            try:
                exec(model.example_code)
            except Exception as e:
                raise e
