import unittest
import os

import numpy as np
from monty import tempfile
import networkx as nx

from pymatgen.util.testing import PymatgenTest
from propnet.core.symbols import Symbol
from propnet.core.exceptions import SymbolConstraintError
from propnet.core.quantity import Quantity
from propnet.core.materials import Material
from propnet.core.graph import Graph
from propnet import ureg


TEST_DIR = os.path.dirname(os.path.abspath(__file__))


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

    def test_get_provenance_graph(self):
        g = Graph()
        qs = [Quantity("bulk_modulus", 100),
              Quantity("shear_modulus", 50),
              Quantity("density", 8.96)]
        mat = Material(qs)
        evaluated = g.evaluate(mat)
        # TODO: this should be tested more thoroughly
        out = list(evaluated['vickers_hardness'])[0]
        with tempfile.ScratchDir('.'):
            out.draw_provenance_graph("out.png")
        pgraph = out.get_provenance_graph()
        end = list(evaluated['vickers_hardness'])[0]
        shortest_lengths = nx.shortest_path_length(pgraph, qs[0])
        self.assertEqual(shortest_lengths[end], 4)

        # This test is useful if one wants to actually make a plot, leaving
        # it in for now
        # from propnet.ext.matproj import MPRester
        # mpr = MPRester()
        # mat = mpr.get_material_for_mpid("mp-66")
        # evaluated = g.evaluate(mat)
        # out = list(evaluated['vickers_hardness'])[-1]
        # out.draw_provenance_graph("out.png", prog='dot')

    def test_nan_checking(self):
        A = Symbol('a', ['A'], ['A'], units='dimensionless', shape=1)
        B = Symbol('b', ['B'], ['B'], units='dimensionless', shape=[2, 2])
        C = Symbol('c', ['C'], ['C'], units='dimensionless', shape=1)
        D = Symbol('d', ['D'], ['D'], units='dimensionless', shape=[2, 2])

        scalar_quantity = Quantity(A, float('nan'))
        non_scalar_quantity = Quantity(B, [[1.0, float('nan')],
                                           [float('nan'), 1.0]])
        complex_scalar_quantity = Quantity(C, complex('nan+nanj'))
        complex_non_scalar_quantity = Quantity(D, [[complex(1.0), complex('nanj')],
                                                   [complex('nan'), complex(1.0)]])

        self.assertTrue(scalar_quantity.contains_nan_value())
        self.assertTrue(non_scalar_quantity.contains_nan_value())
        self.assertTrue(complex_scalar_quantity.contains_nan_value())
        self.assertTrue(complex_non_scalar_quantity.contains_nan_value())

        scalar_quantity = Quantity(A, 1.0)
        non_scalar_quantity = Quantity(B, [[1.0, 2.0],
                                           [2.0, 1.0]])
        complex_scalar_quantity = Quantity(C, complex('1+1j'))
        complex_non_scalar_quantity = Quantity(D, [[complex(1.0), complex('5j')],
                                                   [complex('5'), complex(1.0)]])

        self.assertFalse(scalar_quantity.contains_nan_value())
        self.assertFalse(non_scalar_quantity.contains_nan_value())
        self.assertFalse(complex_scalar_quantity.contains_nan_value())
        self.assertFalse(complex_non_scalar_quantity.contains_nan_value())

    def test_complex_and_imaginary_checking(self):
        A = Symbol('a', ['A'], ['A'], units='dimensionless', shape=1)
        B = Symbol('b', ['B'], ['B'], units='dimensionless', shape=[2, 2])

        real_float_scalar = Quantity(A, 1.0)
        real_float_non_scalar = Quantity(B, [[1.0, 1.0],
                                             [1.0, 1.0]])

        complex_scalar = Quantity(A, complex(1+1j))
        complex_non_scalar = Quantity(B, [[complex(1.0), complex(1.j)],
                                          [complex(1.j), complex(1.0)]])

        complex_scalar_zero_imaginary = Quantity(A, complex(1.0))
        complex_non_scalar_zero_imaginary = Quantity(B, [[complex(1.0), complex(1.0)],
                                                         [complex(1.0), complex(1.0)]])

        complex_scalar_appx_zero_imaginary = Quantity(A, complex(1.0+1e-10j))
        complex_non_scalar_appx_zero_imaginary = Quantity(B, [[complex(1.0), complex(1.0+1e-10j)],
                                                              [complex(1.0+1e-10j), complex(1.0)]])

        self.assertFalse(real_float_scalar.is_complex_type())
        self.assertFalse(real_float_scalar.contains_imaginary_value())
        self.assertFalse(real_float_non_scalar.is_complex_type())
        self.assertFalse(real_float_non_scalar.contains_imaginary_value())

        self.assertTrue(complex_scalar.is_complex_type())
        self.assertTrue(complex_scalar.contains_imaginary_value())
        self.assertTrue(complex_non_scalar.is_complex_type())
        self.assertTrue(complex_non_scalar.contains_imaginary_value())

        self.assertTrue(complex_scalar_zero_imaginary.is_complex_type())
        self.assertFalse(complex_scalar_zero_imaginary.contains_imaginary_value())
        self.assertTrue(complex_non_scalar_zero_imaginary.is_complex_type())
        self.assertFalse(complex_non_scalar_zero_imaginary.contains_imaginary_value())

        self.assertTrue(complex_scalar_appx_zero_imaginary.is_complex_type())
        self.assertFalse(complex_scalar_appx_zero_imaginary.contains_imaginary_value())
        self.assertTrue(complex_non_scalar_appx_zero_imaginary.is_complex_type())
        self.assertFalse(complex_non_scalar_appx_zero_imaginary.contains_imaginary_value())

    def test_numpy_scalar_conversion(self):
        # From custom symbol
        q_int = Quantity(self.custom_symbol, np.int64(5))
        q_float = Quantity(self.custom_symbol, np.float64(5.0))
        q_complex = Quantity(self.custom_symbol, np.complex64(5.0+1.j))

        self.assertTrue(isinstance(q_int.magnitude, int))
        self.assertTrue(isinstance(q_float.magnitude, float))
        self.assertTrue(isinstance(q_complex.magnitude, complex))

        q_int_uncertainty = Quantity(self.custom_symbol, 5, uncertainty=np.int64(1))
        q_float_uncertainty = Quantity(self.custom_symbol, 5.0, uncertainty=np.float64(1.0))
        q_complex_uncertainty = Quantity(self.custom_symbol, 5.0+1j, uncertainty=np.complex64(1.0 + 0.1j))

        self.assertTrue(isinstance(q_int_uncertainty.uncertainty, int))
        self.assertTrue(isinstance(q_float_uncertainty.uncertainty, float))
        self.assertTrue(isinstance(q_complex_uncertainty.uncertainty, complex))


