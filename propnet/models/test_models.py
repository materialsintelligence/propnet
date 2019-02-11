import unittest

import math
import numpy as np

import propnet.models
from propnet.core.registry import Registry


class ModelsTest(unittest.TestCase):
    def test_instantiate_all_models(self):
        models_to_test = []
        for model_name in Registry("models").keys():
            try:
                model = Registry("models").get(model_name)
                self.assertIsNotNone(model)
                models_to_test.append(model_name)
            except Exception as e:
                self.fail('Failed to load model {}: {}'.format(model_name, e))

    def test_model_formatting(self):
        # TODO: clean up tests (self.assertNotNone), test reference format too
        for model in Registry("models").values():
            self.assertIsNotNone(model.name)
            self.assertIsNotNone(model.categories)
            self.assertIsNotNone(model.description)
            self.assertIsNotNone(model.symbol_property_map)
            self.assertIsNotNone(model.implemented_by)
            self.assertNotEqual(model.implemented_by, [])
            self.assertTrue(isinstance(model.symbol_property_map, dict))
            self.assertTrue(len(model.symbol_property_map.keys()) > 0)
            for key in model.symbol_property_map.keys():
                self.assertTrue(isinstance(key, str), 'Invalid symbol_property_map key: ' + str(key))
                self.assertTrue(
                    isinstance(model.symbol_property_map[key], str)
                    and model.symbol_property_map[key] in Registry("symbols").keys())
            self.assertTrue(
                model.connections is not None and isinstance(model.connections, list) and len(model.connections) > 0)
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
        for model in Registry("models").values():
            self.assertTrue(model.validate_from_preset_test())

    def test_example_code_helper(self):

        example_model = Registry("models")['semi_empirical_mobility']

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

        for name, model in Registry("models").items():
            # A little weird, but otherwise you don't get the explicit error
            try:
                exec(model.example_code)
            except Exception as e:
                raise e


if __name__ == "__main__":
    unittest.main()
