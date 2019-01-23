import unittest

from propnet.core.symbols import Symbol
from propnet.symbols import DEFAULT_SYMBOLS


class SymbolTest(unittest.TestCase):
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
            name='youngs_modulus',#
            units= [1.0, [["gigapascal", 1.0]]], #ureg.parse_expression("GPa"),
            display_names=["Young's modulus", "Elastic modulus"],
            display_symbols=["E"],
            shape=1,
            comment=""
        )

        self.assertEqual(sample_symbol_type,
                         Symbol.from_dict(sample_symbol_type_dict))

    def test_property_formatting(self):
        """
        Goes through the Quantity .yaml files and ensures the definitions are complete.
        """
        for st in DEFAULT_SYMBOLS.values():
            self.assertTrue(st.name is not None and st.name.isidentifier())
            self.assertTrue(st.category is not None and st.category
                            in ('property', 'condition', 'object'))
            self.assertTrue(st.display_names is not None and isinstance(st.display_names, list) and
                            len(st.display_names) != 0)
            self.assertTrue(st.display_symbols is not None and isinstance(st.display_symbols, list) and
                            len(st.display_symbols) != 0, st.name)
            self.assertTrue(st.comment is not None and isinstance(st.comment, str))
            if st.category != 'object':
                self.assertIsNotNone(st.units)

    def test_all_properties(self):
        self.assertEqual(str(DEFAULT_SYMBOLS['density'].units),
                         '1.0 gram / centimeter ** 3')
