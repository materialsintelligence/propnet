import unittest

# noinspection PyUnresolvedReferences
import propnet.models
from propnet.core.registry import Registry


class DefaultModelsTest(unittest.TestCase):
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
            if not model.is_builtin:
                continue
            self.assertIsNotNone(model.name)
            msg = "{} has invalid property".format(model.name)
            self.assertIsNotNone(model.categories, msg=msg)
            self.assertIsNotNone(model.description, msg=msg)
            self.assertIsNotNone(model.symbol_property_map, msg=msg)
            self.assertIsNotNone(model.implemented_by, msg=msg)
            self.assertNotEqual(model.implemented_by, [], msg=msg)
            self.assertTrue(isinstance(model.symbol_property_map, dict), msg=msg)
            self.assertTrue(len(model.symbol_property_map.keys()) > 0, msg=msg)
            for key in model.symbol_property_map.keys():
                self.assertTrue(isinstance(key, str), 'Invalid symbol_property_map key: ' + str(key))
                self.assertTrue(
                    isinstance(model.symbol_property_map[key], str)
                    and model.symbol_property_map[key] in Registry("symbols").keys(), msg=msg)
            self.assertTrue(
                model.connections is not None and isinstance(model.connections, list) and len(model.connections) > 0,
                msg=msg)
            for reference in model.references:
                self.assertTrue(reference.startswith('@'), msg=msg)
            for item in model.connections:
                self.assertIsNotNone(item, msg=msg)
                self.assertTrue(isinstance(item, dict), msg=msg)
                self.assertTrue('inputs' in item.keys(), msg=msg)
                self.assertTrue('outputs' in item.keys(), msg=msg)
                self.assertIsNotNone(item['inputs'], msg=msg)
                self.assertIsNotNone(item['outputs'], msg=msg)
                self.assertTrue(isinstance(item['inputs'], list), msg=msg)
                self.assertTrue(isinstance(item['outputs'], list), msg=msg)
                self.assertTrue(len(item['inputs']) > 0, msg=msg)
                self.assertTrue(len(item['outputs']) > 0, msg=msg)
                for in_symb in item['inputs']:
                    self.assertIsNotNone(in_symb, msg=msg)
                    self.assertTrue(isinstance(in_symb, str), msg=msg)
                    self.assertTrue(in_symb in model.symbol_property_map.keys(), msg=msg)
                for out_symb in item['outputs']:
                    self.assertIsNotNone(out_symb, msg=msg)
                    self.assertIsNotNone(isinstance(out_symb, str), msg=msg)
                    self.assertTrue(out_symb in model.symbol_property_map.keys(), msg=msg)

    def test_validate_all_models(self):
        for model in Registry("models").values():
            if model._test_data is not None:
                self.assertTrue(model.validate_from_preset_test(), msg="{} model failed".format(model.name))
            else:
                self.assertFalse(model.is_builtin, msg="{} is a built-in model and "
                                                       "contains no test data".format(model.name))

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

        for model in Registry("models").values():
            # A little weird, but otherwise you don't get the explicit error
            if model.is_builtin:
                try:
                    exec(model.example_code)
                except Exception as e:
                    raise e


if __name__ == "__main__":
    unittest.main()
