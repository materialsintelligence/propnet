import unittest
from propnet.core.properties import *


class PropertiesTest(unittest.TestCase):

    def testPropertyMetadata(self):

        sample_property_type_dict = {
            'name': 'youngs_modulus',
            'units': [1.0, [["gigapascal", 1.0]]],
            'display_names': ["Young's modulus", "Elastic modulus"],
            'display_symbols': ["E"],
            'dimension': 1,
            'test_value': 130.0,
            'comment': ""
        }

        sample_property_type = PropertyMetadata(
            name='youngs_modulus',
            units= [1.0, [["gigapascal", 1.0]]], #ureg.parse_expression("GPa"),
            display_names=["Young's modulus", "Elastic modulus"],
            display_symbols=["E"],
            dimension=1,
            test_value=130.0,
            comment=""
        )

        self.assertEqual(sample_property_type,
                         PropertyMetadata.from_dict(sample_property_type_dict))