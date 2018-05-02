import unittest

from propnet.core.symbols import *
from propnet import ureg

from propnet.symbols import DEFAULT_SYMBOL_TYPES


class PropertiesTest(unittest.TestCase):

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
        for st in DEFAULT_SYMBOL_TYPES.values():
            self.assertTrue(st.name is not None and st.name.isidentifier())
            self.assertTrue(st.category is not None and st.category in ('property', 'condition', 'object'))
            self.assertTrue(st.units is not None and type(st.units) == ureg.Quantity)
            self.assertTrue(st.display_names is not None and isinstance(st.display_names, list) and
                            len(st.display_names) != 0)
            self.assertTrue(st.display_symbols is not None and isinstance(st.display_symbols, list) and
                            len(st.display_symbols) != 0)
            self.assertTrue(st.dimension is not None)
            self.assertTrue(st.comment is not None and isinstance(st.comment, str))

    def test_all_properties(self):
        self.assertEqual(str(DEFAULT_SYMBOL_TYPES['density'].units),
                         '1.0 gram / centimeter ** 3')

    #def test_serialization(self):
#
    #    symbol = Quantity('band_gap', 3.0)
#
    #    self.assertDictEqual(symbol.as_dict(), {})