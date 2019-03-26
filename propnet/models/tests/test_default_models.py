import unittest

# noinspection PyUnresolvedReferences
from propnet.models import add_builtin_models_to_registry
from propnet.core.registry import Registry
from collections import defaultdict


class DefaultModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Registry.clear_all_registries()
        add_builtin_models_to_registry()

    def test_models_have_unique_names(self):
        add_builtin_models_to_registry()
        from propnet.models.serialized import _EQUATION_MODEL_NAMES_LIST
        serialized_model_names_set = set(_EQUATION_MODEL_NAMES_LIST)
        serialized_model_names_list = _EQUATION_MODEL_NAMES_LIST.copy()
        for name in serialized_model_names_set:
            serialized_model_names_list.remove(name)
        self.assertTrue(len(serialized_model_names_list) == 0,
                        msg="Two or more equation models have the "
                            "same name(s): {}".format(serialized_model_names_list))

        from propnet.models.python import _PYTHON_MODEL_NAMES_LIST
        python_model_names_set = set(_PYTHON_MODEL_NAMES_LIST)
        python_model_names_list = _PYTHON_MODEL_NAMES_LIST.copy()
        for name in python_model_names_set:
            python_model_names_list.remove(name)
        self.assertTrue(len(python_model_names_list) == 0,
                        msg="Two or more python models have the "
                            "same name(s): {}".format(python_model_names_list))

        from propnet.models.composite import _COMPOSITE_MODEL_NAMES_LIST
        composite_model_names_set = set(_COMPOSITE_MODEL_NAMES_LIST)
        composite_model_names_list = _COMPOSITE_MODEL_NAMES_LIST.copy()
        for name in composite_model_names_set:
            composite_model_names_list.remove(name)
        self.assertTrue(len(composite_model_names_list) == 0,
                        msg="Two or more composite models have the "
                            "same name(s): {}".format(composite_model_names_list))

        all_models_set = set.union(serialized_model_names_set,
                                   python_model_names_set,
                                   composite_model_names_set)
        all_models_list = \
            _EQUATION_MODEL_NAMES_LIST + \
            _PYTHON_MODEL_NAMES_LIST + \
            _COMPOSITE_MODEL_NAMES_LIST
        for name in all_models_set:
            all_models_list.remove(name)

        sources = defaultdict(list)
        for name in all_models_list:
            for model_type, name_list in zip(('serialized', 'python', 'composite'),
                                             (_EQUATION_MODEL_NAMES_LIST,
                                              _PYTHON_MODEL_NAMES_LIST,
                                              _COMPOSITE_MODEL_NAMES_LIST)):
                if name in name_list:
                    sources[name].append(model_type)
        self.assertTrue(len(all_models_list) == 0,
                        msg="Two or more models of different types"
                            " have the same name {{name: [model types]}}:\n{}".format(dict(sources)))

    def test_reimport_models(self):
        Registry("models").pop('debye_temperature')
        Registry("models").pop('is_metallic')
        Registry("composite_models").pop('pilling_bedworth_ratio')
        self.assertNotIn("debye_temperature", Registry("models"))
        self.assertNotIn("is_metallic", Registry("models"))
        self.assertNotIn("pilling_bedworth_ratio", Registry("composite_models"))
        from propnet.models.serialized import add_builtin_models_to_registry as serialized_add
        serialized_add()
        self.assertIn("debye_temperature", Registry("models"))
        self.assertNotIn("is_metallic", Registry("models"))
        self.assertNotIn("pilling_bedworth_ratio", Registry("composite_models"))
        from propnet.models.python import add_builtin_models_to_registry as python_add
        python_add()
        self.assertIn("debye_temperature", Registry("models"))
        self.assertIn("is_metallic", Registry("models"))
        self.assertNotIn("pilling_bedworth_ratio", Registry("composite_models"))
        from propnet.models.composite import add_builtin_models_to_registry as composite_add
        composite_add()
        self.assertIn("debye_temperature", Registry("models"))
        self.assertIn("is_metallic", Registry("models"))
        self.assertIn("pilling_bedworth_ratio", Registry("composite_models"))

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
            self.assertIsNotNone(model.variable_symbol_map, msg=msg)
            self.assertIsNotNone(model.implemented_by, msg=msg)
            self.assertNotEqual(model.implemented_by, [], msg=msg)
            self.assertTrue(isinstance(model.variable_symbol_map, dict), msg=msg)
            self.assertTrue(len(model.variable_symbol_map.keys()) > 0, msg=msg)
            for key in model.variable_symbol_map.keys():
                self.assertTrue(isinstance(key, str), 'Invalid variable_symbol_map key: ' + str(key))
                self.assertTrue(
                    isinstance(model.variable_symbol_map[key], str)
                    and model.variable_symbol_map[key] in Registry("symbols").keys(), msg=msg)
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
                    self.assertTrue(in_symb in model.variable_symbol_map.keys(), msg=msg)
                for out_symb in item['outputs']:
                    self.assertIsNotNone(out_symb, msg=msg)
                    self.assertIsNotNone(isinstance(out_symb, str), msg=msg)
                    self.assertTrue(out_symb in model.variable_symbol_map.keys(), msg=msg)

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
