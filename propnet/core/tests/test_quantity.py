import unittest
import os

import numpy as np
from monty import tempfile
import networkx as nx
import copy

from pymatgen.util.testing import PymatgenTest
from propnet.core.symbols import Symbol
from propnet.symbols import DEFAULT_SYMBOLS
from propnet.core.exceptions import SymbolConstraintError
from propnet.core.quantity import QuantityFactory, NumQuantity, ObjQuantity
from propnet.core.materials import Material
from propnet.core.graph import Graph
from propnet import ureg

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


class QuantityTest(unittest.TestCase):
    def setUp(self):
        self.custom_symbol = Symbol("A", units='dimensionless')
        self.constraint_symbol = Symbol("A", constraint="A > 0",
                                        units='dimensionless')
        self.custom_object_symbol = Symbol("B", category='object')

    def test_quantity_construction(self):
        # From custom symbol
        q = QuantityFactory.create_quantity(self.custom_symbol, 5.0)
        self.assertEqual(q.value.magnitude, 5.0)
        self.assertIsInstance(q.value, ureg.Quantity)
        # From canonical symbol
        q = QuantityFactory.create_quantity("bulk_modulus", 100)
        self.assertEqual(q.value.magnitude, 100)
        # From custom symbol with constraint
        with self.assertRaises(SymbolConstraintError):
            QuantityFactory.create_quantity(self.constraint_symbol, -500)
        # From canonical symbol with constraint
        with self.assertRaises(SymbolConstraintError):
            QuantityFactory.create_quantity("bulk_modulus", -500)

    def test_from_default(self):
        default = QuantityFactory.from_default('temperature')
        self.assertEqual(default, QuantityFactory.create_quantity('temperature', 300))
        default = QuantityFactory.from_default('relative_permeability')
        self.assertEqual(default, QuantityFactory.create_quantity("relative_permeability", 1))

    def test_from_weighted_mean(self):
        qlist = [QuantityFactory.create_quantity(self.custom_symbol, val)
                 for val in np.arange(1, 2.01, 0.01)]
        qagg = NumQuantity.from_weighted_mean(qlist)
        self.assertAlmostEqual(qagg.magnitude, 1.5)
        self.assertAlmostEqual(qagg.uncertainty, 0.2915475947422652)

    def test_is_cyclic(self):
        # Simple test
        pass

    def test_pretty_string(self):
        quantity = QuantityFactory.create_quantity('bulk_modulus', 100)
        self.assertEqual(quantity.pretty_string(3), "100 GPa")

    def test_to(self):
        quantity = QuantityFactory.create_quantity('band_gap', 3.0, 'eV')
        new = quantity.to('joules')
        self.assertEqual(new.magnitude, 4.80652959e-19)
        self.assertEqual(new.units, 'joule')

    def test_properties(self):
        # Test units, magnitude
        q = QuantityFactory.create_quantity("bulk_modulus", 100)
        self.assertIsInstance(q, NumQuantity)
        self.assertEqual(q.units, "gigapascal")
        self.assertEqual(q.magnitude, 100)

        # Ensure non-pint values raise error with units, magnitude
        structure = PymatgenTest.get_structure('Si')
        q = QuantityFactory.create_quantity("structure", structure)
        self.assertIsInstance(q, ObjQuantity)

    def test_get_provenance_graph(self):
        g = Graph()
        qs = [QuantityFactory.create_quantity("bulk_modulus", 100),
              QuantityFactory.create_quantity("shear_modulus", 50),
              QuantityFactory.create_quantity("density", 8.96)]
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
        E = Symbol('e', ['E'], ['E'], category='object', shape=1)

        scalar_quantity = QuantityFactory.create_quantity(A, float('nan'))
        non_scalar_quantity = QuantityFactory.create_quantity(B, [[1.0, float('nan')],
                                                                  [float('nan'), 1.0]])
        complex_scalar_quantity = QuantityFactory.create_quantity(C, complex('nan+nanj'))
        complex_non_scalar_quantity = QuantityFactory.create_quantity(D, [[complex(1.0), complex('nanj')],
                                                                          [complex('nan'), complex(1.0)]])

        self.assertTrue(scalar_quantity.contains_nan_value())
        self.assertTrue(non_scalar_quantity.contains_nan_value())
        self.assertTrue(complex_scalar_quantity.contains_nan_value())
        self.assertTrue(complex_non_scalar_quantity.contains_nan_value())

        scalar_quantity = QuantityFactory.create_quantity(A, 1.0)
        non_scalar_quantity = QuantityFactory.create_quantity(B, [[1.0, 2.0],
                                                                  [2.0, 1.0]])
        complex_scalar_quantity = QuantityFactory.create_quantity(C, complex('1+1j'))
        complex_non_scalar_quantity = QuantityFactory.create_quantity(D, [[complex(1.0), complex('5j')],
                                                                          [complex('5'), complex(1.0)]])

        self.assertFalse(scalar_quantity.contains_nan_value())
        self.assertFalse(non_scalar_quantity.contains_nan_value())
        self.assertFalse(complex_scalar_quantity.contains_nan_value())
        self.assertFalse(complex_non_scalar_quantity.contains_nan_value())

        non_numerical = QuantityFactory.create_quantity(E, 'test')
        self.assertFalse(non_numerical.contains_nan_value())

    def test_complex_and_imaginary_checking(self):
        A = Symbol('a', ['A'], ['A'], units='dimensionless', shape=1)
        B = Symbol('b', ['B'], ['B'], units='dimensionless', shape=[2, 2])
        # TODO: Revisit this when splitting quantity class into non-numerical and numerical
        C = Symbol('c', ['C'], ['C'], category='object', shape=1)

        real_float_scalar = QuantityFactory.create_quantity(A, 1.0)
        real_float_non_scalar = QuantityFactory.create_quantity(B, [[1.0, 1.0],
                                                                    [1.0, 1.0]])

        complex_scalar = QuantityFactory.create_quantity(A, complex(1 + 1j))
        complex_non_scalar = QuantityFactory.create_quantity(B, [[complex(1.0), complex(1.j)],
                                                                 [complex(1.j), complex(1.0)]])

        complex_scalar_zero_imaginary = QuantityFactory.create_quantity(A, complex(1.0))
        complex_non_scalar_zero_imaginary = QuantityFactory.create_quantity(B, [[complex(1.0), complex(1.0)],
                                                                                [complex(1.0), complex(1.0)]])

        complex_scalar_appx_zero_imaginary = QuantityFactory.create_quantity(A, complex(1.0 + 1e-10j))
        complex_non_scalar_appx_zero_imaginary = QuantityFactory.create_quantity(B,
                                                                                 [[complex(1.0), complex(1.0 + 1e-10j)],
                                                                                  [complex(1.0 + 1e-10j),
                                                                                   complex(1.0)]])

        non_numerical = QuantityFactory.create_quantity(C, 'test')

        # Test is_complex_type() with...
        # ...Quantity objects
        self.assertFalse(NumQuantity.is_complex_type(real_float_scalar))
        self.assertFalse(NumQuantity.is_complex_type(real_float_non_scalar))
        self.assertTrue(NumQuantity.is_complex_type(complex_scalar))
        self.assertTrue(NumQuantity.is_complex_type(complex_non_scalar))
        self.assertTrue(NumQuantity.is_complex_type(complex_scalar_zero_imaginary))
        self.assertTrue(NumQuantity.is_complex_type(complex_non_scalar_zero_imaginary))
        self.assertTrue(NumQuantity.is_complex_type(complex_scalar_appx_zero_imaginary))
        self.assertTrue(NumQuantity.is_complex_type(complex_non_scalar_appx_zero_imaginary))
        self.assertFalse(NumQuantity.is_complex_type(non_numerical))

        # ...primitive types
        self.assertFalse(NumQuantity.is_complex_type(1))
        self.assertFalse(NumQuantity.is_complex_type(1.))
        self.assertTrue(NumQuantity.is_complex_type(1j))
        self.assertFalse(NumQuantity.is_complex_type('test'))

        # ...np.array types
        self.assertFalse(NumQuantity.is_complex_type(np.array([1])))
        self.assertFalse(NumQuantity.is_complex_type(np.array([1.])))
        self.assertTrue(NumQuantity.is_complex_type(np.array([1j])))
        self.assertFalse(NumQuantity.is_complex_type(np.array(['test'])))

        # ...ureg Quantity objects
        self.assertFalse(NumQuantity.is_complex_type(ureg.Quantity(1)))
        self.assertFalse(NumQuantity.is_complex_type(ureg.Quantity(1.)))
        self.assertTrue(NumQuantity.is_complex_type(ureg.Quantity(1j)))
        self.assertFalse(NumQuantity.is_complex_type(ureg.Quantity([1])))
        self.assertFalse(NumQuantity.is_complex_type(ureg.Quantity([1.])))
        self.assertTrue(NumQuantity.is_complex_type(ureg.Quantity([1j])))

        # Check member functions
        self.assertFalse(real_float_scalar.contains_complex_type())
        self.assertFalse(real_float_scalar.contains_imaginary_value())
        self.assertFalse(real_float_non_scalar.contains_complex_type())
        self.assertFalse(real_float_non_scalar.contains_imaginary_value())

        self.assertTrue(complex_scalar.contains_complex_type())
        self.assertTrue(complex_scalar.contains_imaginary_value())
        self.assertTrue(complex_non_scalar.contains_complex_type())
        self.assertTrue(complex_non_scalar.contains_imaginary_value())

        self.assertTrue(complex_scalar_zero_imaginary.contains_complex_type())
        self.assertFalse(complex_scalar_zero_imaginary.contains_imaginary_value())
        self.assertTrue(complex_non_scalar_zero_imaginary.contains_complex_type())
        self.assertFalse(complex_non_scalar_zero_imaginary.contains_imaginary_value())

        self.assertTrue(complex_scalar_appx_zero_imaginary.contains_complex_type())
        self.assertFalse(complex_scalar_appx_zero_imaginary.contains_imaginary_value())
        self.assertTrue(complex_non_scalar_appx_zero_imaginary.contains_complex_type())
        self.assertFalse(complex_non_scalar_appx_zero_imaginary.contains_imaginary_value())

        self.assertFalse(non_numerical.contains_complex_type())
        self.assertFalse(non_numerical.contains_imaginary_value())

    def test_numpy_scalar_conversion(self):
        # From custom symbol
        q_int = QuantityFactory.create_quantity(self.custom_symbol, np.int64(5))
        q_float = QuantityFactory.create_quantity(self.custom_symbol, np.float64(5.0))
        q_complex = QuantityFactory.create_quantity(self.custom_symbol, np.complex64(5.0 + 1.j))

        self.assertTrue(isinstance(q_int.magnitude, int))
        self.assertTrue(isinstance(q_float.magnitude, float))
        self.assertTrue(isinstance(q_complex.magnitude, complex))

        q_int_uncertainty = QuantityFactory.create_quantity(self.custom_symbol, 5, uncertainty=np.int64(1))
        q_float_uncertainty = QuantityFactory.create_quantity(self.custom_symbol, 5.0, uncertainty=np.float64(1.0))
        q_complex_uncertainty = QuantityFactory.create_quantity(self.custom_symbol, 5.0 + 1j,
                                                                uncertainty=np.complex64(1.0 + 0.1j))

        self.assertTrue(isinstance(q_int_uncertainty.uncertainty.magnitude, int))
        self.assertTrue(isinstance(q_float_uncertainty.uncertainty.magnitude, float))
        self.assertTrue(isinstance(q_complex_uncertainty.uncertainty.magnitude, complex))

    def test_as_dict_from_dict(self):
        q = QuantityFactory.create_quantity(self.custom_symbol, 5, tags='experimental', uncertainty=1)
        d = q.as_dict()

        self.assertDictEqual(d, {"@module": "propnet.core.quantity",
                                 "@class": "NumQuantity",
                                 "value": 5,
                                 "units": "dimensionless",
                                 "provenance": q.provenance.as_dict(),
                                 "symbol_type": self.custom_symbol.as_dict(),
                                 "tags": 'experimental',
                                 "uncertainty": (1, ())
                                 })

        q_from = QuantityFactory.from_dict(d)

        self.assertIsInstance(q_from, NumQuantity)
        self.assertEqual(q_from.symbol, q.symbol)
        self.assertEqual(q_from.value, q.value)
        self.assertEqual(q_from.units, q.units)
        self.assertEqual(q_from.tags, q.tags)
        self.assertEqual(q_from.uncertainty, q.uncertainty)
        self.assertEqual(q_from.provenance, q.provenance)

        q_from = NumQuantity.from_dict(d)

        self.assertIsInstance(q_from, NumQuantity)
        self.assertEqual(q_from.symbol, q.symbol)
        self.assertEqual(q_from.value, q.value)
        self.assertEqual(q_from.units, q.units)
        self.assertEqual(q_from.tags, q.tags)
        self.assertEqual(q_from.uncertainty, q.uncertainty)
        self.assertEqual(q_from.provenance, q.provenance)

        q = QuantityFactory.create_quantity(DEFAULT_SYMBOLS['debye_temperature'],
                                            500, tags='experimental', uncertainty=10)
        d = q.as_dict()

        self.assertDictEqual(d, {"@module": "propnet.core.quantity",
                                 "@class": "NumQuantity",
                                 "value": 500,
                                 "units": "kelvin",
                                 "provenance": q.provenance.as_dict(),
                                 "symbol_type": "debye_temperature",
                                 "tags": 'experimental',
                                 "uncertainty": (10, (('kelvin', 1.0),))
                                 })

        q_from = QuantityFactory.from_dict(d)

        self.assertIsInstance(q_from, NumQuantity)
        self.assertEqual(q_from.symbol, q.symbol)
        self.assertEqual(q_from.value, q.value)
        self.assertEqual(q_from.units, q.units)
        self.assertEqual(q_from.tags, q.tags)
        self.assertEqual(q_from.uncertainty, q.uncertainty)
        self.assertEqual(q_from.provenance, q.provenance)

        q_from = NumQuantity.from_dict(d)

        self.assertIsInstance(q_from, NumQuantity)
        self.assertEqual(q_from.symbol, q.symbol)
        self.assertEqual(q_from.value, q.value)
        self.assertEqual(q_from.units, q.units)
        self.assertEqual(q_from.tags, q.tags)
        self.assertEqual(q_from.uncertainty, q.uncertainty)
        self.assertEqual(q_from.provenance, q.provenance)

    def test_equality(self):
        q1 = QuantityFactory.create_quantity(self.custom_symbol, 5, tags='experimental', uncertainty=1)
        q1_copy = copy.deepcopy(q1)
        q2 = QuantityFactory.create_quantity(self.custom_symbol, 6, tags='experimental', uncertainty=2)

        self.assertEqual(q1, q1_copy)
        self.assertEqual(q1.symbol, q1_copy.symbol)
        self.assertEqual(q1.value, q1_copy.value)
        self.assertEqual(q1.units, q1_copy.units)
        self.assertEqual(q1.tags, q1_copy.tags)
        self.assertEqual(q1.uncertainty, q1_copy.uncertainty)
        self.assertEqual(q1.provenance, q1_copy.provenance)

        self.assertNotEqual(q1, q2)

        fields = list(q1.__dict__.keys())

        # This is to check to see if we modified the fields in the object, in case we need to add
        # to our equality statement
        self.assertListEqual(fields, ['_value', '_symbol_type',
                                      '_tags', '_provenance',
                                      '_internal_id', '_uncertainty'])

