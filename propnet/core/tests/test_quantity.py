import unittest

import numpy as np

from propnet.core.quantity import Quantity
from propnet.core.symbols import Symbol
from propnet.core.exceptions import SymbolConstraintError
from propnet import ureg


class QuantityTest(unittest.TestCase):
    def setUp(self):
        self.custom_symbol = Symbol("A", units='dimensionless')
        self.constraint_symbol = Symbol("A", constraint="A > 0",
                                        units='dimensionless')

    def test_quantity_construction(self):
        # From custom symbol
        q = Quantity(self.custom_symbol, 5.0)
        self.assertEqual(q.value.magnitude, 5.0)
        self.assertIsInstance(q.value, ureg.Quantity)
        # From canonical symbol
        q = Quantity("bulk_modulus", 100)
        self.assertEqual(q.value.magnitude, 100)
        # From custom symbol with constraint
        with self.assertRaises(SymbolConstraintError):
            Quantity(self.constraint_symbol, -500)
        # From canonical symbol with constraint
        with self.assertRaises(SymbolConstraintError):
            Quantity("bulk_modulus", -500)

    def test_from_weighted_mean(self):
        qlist = [Quantity(self.custom_symbol, val)
                 for val in np.arange(1, 2.01, 0.01)]
        qagg = Quantity.from_weighted_mean(qlist)
        self.assertEqual(qagg.value.magnitude, 1.5)
        self.assertEqual(qagg.value.std_dev, 0.25)

    def test_is_cyclic(self):
        # Simple test
        pass
