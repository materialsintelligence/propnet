import unittest
from propnet.core.symbols import *

# these are properties distributed with Propnet as
# yaml files in the properties folder
from propnet.symbols import *

class PropertiesTest(unittest.TestCase):
    """ """

    def testPropertyMetadata(self):
        """ """

        sample_symbol_type_dict = {
            'name': 'youngs_modulus',
            'units': [1.0, [["gigapascal", 1.0]]],
            'display_names': ["Young's modulus", "Elastic modulus"],
            'display_symbols': ["E"],
            'dimension': 1,
            'test_value': 130.0,
            'comment': ""
        }

        sample_symbol_type = SymbolMetadata(
            name='youngs_modulus',
            units= [1.0, [["gigapascal", 1.0]]], #ureg.parse_expression("GPa"),
            display_names=["Young's modulus", "Elastic modulus"],
            display_symbols=["E"],
            dimension=1,
            test_value=130.0,
            comment=""
        )

        self.assertEqual(sample_symbol_type,
                         SymbolMetadata.from_dict(sample_symbol_type_dict))

    def testAllProperties(self):
        """ """

        all_properties = {name: SymbolType[name] for name in all_symbol_names}

        self.assertEqual(str(all_properties['density'].value.units), '1.0 gram / centimeter ** 3')