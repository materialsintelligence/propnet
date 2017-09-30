import unittest
from propnet.core.properties import *

class PropertiesTest(unittest.TestCase):

    def testLoader(self):

        sample_property_type_dict = {
            'name': "youngs_modulus",
            'units': [["gigapascal", 1.0]],
            'display_names': ["Young's modulus", "Elastic modulus"],
            'display_symbols': ["E"],
            'dimension': 1,
            'test_value': 130.0,
            'comment': ""
        }

        sample_property_type = PropertyType(
            name='youngs_modulus',
            units=ureg.parse_expression("GPa"),
            display_names=["Young's modulus", "Elastic modulus"],
            display_symbols=["E"],
            dimension=1,
            test_value=130.0,
            comment=""
        )

        loader = PropertyLoader([sample_property_type_dict])

        self.assertEqual(sample_property_type, loader.properties[0])