import unittest

import numpy as np

from pymatgen.util.testing import PymatgenTest
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

    def test_from_default(self):
        default = Quantity.from_default('temperature')
        self.assertEqual(default, Quantity('temperature', 300))
        default = Quantity.from_default('relative_permeability')
        self.assertEqual(default, Quantity("relative_permeability", 1))

    def test_from_weighted_mean(self):
        qlist = [Quantity(self.custom_symbol, val)
                 for val in np.arange(1, 2.01, 0.01)]
        qagg = Quantity.from_weighted_mean(qlist)
        self.assertAlmostEqual(qagg.magnitude, 1.5)
        self.assertAlmostEqual(qagg.uncertainty, 0.2915475947422652)

    def test_is_cyclic(self):
        # Simple test
        pass

    def test_pretty_string(self):
        quantity = Quantity('bulk_modulus', 100)
        self.assertEqual(quantity.pretty_string(3), "100 GPa")

    def test_to(self):
        quantity = Quantity('band_gap', 3.0, 'eV')
        new = quantity.to('joules')
        self.assertEqual(new.magnitude, 4.80652959e-19)
        self.assertEqual(new.units, 'joule')

    def test_properties(self):
        # Test units, magnitude
        q = Quantity("bulk_modulus", 100)
        self.assertEqual(q.units, "gigapascal")
        self.assertEqual(q.magnitude, 100)

        # Ensure non-pint values raise error with units, magnitude
        structure = PymatgenTest.get_structure('Si')
        q = Quantity("structure", structure)
        with self.assertRaises(ValueError):
            print(q.units)
        with self.assertRaises(ValueError):
            print(q.magnitude)
