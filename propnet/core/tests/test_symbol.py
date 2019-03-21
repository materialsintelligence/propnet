import unittest

from propnet.core.symbols import Symbol
from propnet.core.registry import Registry


class SymbolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Registry.clear_all_registries()

    @classmethod
    def tearDownClass(cls):
        Registry.clear_all_registries()

    def test_property_construction(self):
        sample_symbol_type_dict = {
            'name': 'youngs_modulus',
            'units': [1.0, [["gigapascal", 1.0]]],
            'display_names': ["Young's modulus", "Elastic modulus"],
            'display_symbols': ["E"],
            'shape': 1,
            'comment': ""
        }

        sample_symbol_type = Symbol(
            name='youngs_modulus',  #
            units=[1.0, [["gigapascal", 1.0]]],
            display_names=["Young's modulus", "Elastic modulus"],
            display_symbols=["E"],
            shape=1,
            comment="")

        self.assertEqual(sample_symbol_type,
                         Symbol.from_dict(sample_symbol_type_dict))

    def test_symbol_register_unregister(self):
        A = Symbol('a', ['A'], ['A'], units='dimensionless', shape=1)

        self.assertIn(A.name, Registry("symbols"))
        self.assertTrue(A.registered)
        A.unregister()
        self.assertNotIn(A.name, Registry("symbols"))
        self.assertFalse(A.registered)
        A.register()
        self.assertTrue(A.registered)
        with self.assertRaises(KeyError):
            A.register(overwrite_registry=False)

        A.unregister()
        A = Symbol('a', ['A'], ['A'], units='dimensionless', shape=1,
                   register=False)
        self.assertNotIn(A.name, Registry("symbols"))
        self.assertFalse(A.registered)

        A.register()
        with self.assertRaises(KeyError):
            _ = Symbol('a', ['A'], ['A'], units='dimensionless', shape=1,
                       register=True, overwrite_registry=False)

        A_replacement = Symbol('a', ['A^*'], ['A^*'], units='kilogram', shape=1)

        A_registered = Registry("symbols")['a']
        self.assertIs(A_registered, A_replacement)
        self.assertIsNot(A_registered, A)


if __name__ == "__main__":
    unittest.main()
