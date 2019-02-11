import unittest

import math
import numpy as np

import propnet.models
import propnet.symbols

from propnet.core.models import EquationModel
from propnet.core.symbols import Symbol
from propnet.core.quantity import QuantityFactory
from propnet.core.registry import Registry

# TODO: test PyModule, PyModel
# TODO: separate these into specific tests of model functionality
#       and validation of default models

class ModelTest(unittest.TestCase):

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
        out = model.evaluate({'l1': QuantityFactory.create_quantity(L, 1, 'meter'),
                              'l2': QuantityFactory.create_quantity(L, 2)}, allow_failure=False)

        self.assertTrue(math.isclose(out['a'].magnitude, 200.0))
        self.assertTrue(out['a'].units == A.units)

    def test_model_returns_nan(self):
        # This tests model failure with scalar nan.
        # Quantity class has other more thorough tests.

        A = Symbol('a', ['A'], ['A'], units='dimensionless', shape=1)
        B = Symbol('b', ['B'], ['B'], units='dimensionless', shape=1)
        get_config = {
            'name': 'equality',
            # 'connections': [{'inputs': ['b'], 'outputs': ['a']}],
            'equations': ['a = b'],
            # 'unit_map': {'a': "dimensionless", 'a': "dimensionless"}
            'symbol_property_map': {"a": A, "b": B}
        }
        model = EquationModel(**get_config)
        out = model.evaluate({'b': QuantityFactory.create_quantity(B, float('nan'))},
                             allow_failure=True)
        self.assertFalse(out['successful'])
        self.assertEqual(out['message'], 'Evaluation returned invalid values (NaN)')

    def test_model_returns_complex(self):
        # This tests model failure with scalar complex.
        # Quantity class has other more thorough tests.

        A = Symbol('a', ['A'], ['A'], units='dimensionless', shape=1)
        B = Symbol('b', ['B'], ['B'], units='dimensionless', shape=1)
        get_config = {
            'name': 'add_complex_value',
            # 'connections': [{'inputs': ['b'], 'outputs': ['a']}],
            'equations': ['a = b + 1j'],
            # 'unit_map': {'a': "dimensionless", 'a': "dimensionless"}
            'symbol_property_map': {"a": A, "b": B}
        }
        model = EquationModel(**get_config)
        out = model.evaluate({'b': QuantityFactory.create_quantity(B, 5)},
                             allow_failure=True)
        self.assertFalse(out['successful'])
        self.assertEqual(out['message'], 'Evaluation returned invalid values (complex)')

        out = model.evaluate({'b': QuantityFactory.create_quantity(B, 5j)},
                             allow_failure=True)
        self.assertTrue(out['successful'])
        self.assertTrue(np.isclose(out['a'].magnitude, 6j))

if __name__ == "__main__":
    unittest.main()
