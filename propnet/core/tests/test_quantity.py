import unittest
import os

import numpy as np
from monty import tempfile
import networkx as nx
import copy
import logging

from pymatgen.util.testing import PymatgenTest
from propnet.core.symbols import Symbol
from propnet.core.exceptions import SymbolConstraintError
from propnet.core.quantity import QuantityFactory, NumQuantity, ObjQuantity
from propnet.core.materials import Material
from propnet.core.graph import Graph
from propnet import ureg
from propnet.core.provenance import ProvenanceElement
from propnet.core.utils import LogSniffer

# noinspection PyUnresolvedReferences
import propnet.symbols
from propnet.core.registry import Registry

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


class QuantityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.custom_symbol = Symbol("A", units='dimensionless')
        cls.constraint_symbol = Symbol("A", constraint="A > 0",
                                        units='dimensionless')
        cls.custom_object_symbol = Symbol("B", category='object')
        cls.maxDiff = None

    @classmethod
    def tearDownClass(cls):
        non_builtin_syms = [k for k, v in Registry("symbols").items() if not v.is_builtin]
        for sym in non_builtin_syms:
            Registry("symbols").pop(sym)
            Registry("units").pop(sym)

    def test_quantity_construction(self):
        # From custom numerical symbol
        q = QuantityFactory.create_quantity(self.custom_symbol, 5.0)
        self.assertIsInstance(q, NumQuantity)
        self.assertIsInstance(q.value, ureg.Quantity)
        self.assertAlmostEqual(q.value.magnitude, 5.0)
        self.assertEqual(q.value.units.format_babel(), 'dimensionless')
        # From canonical numerical symbol
        q = QuantityFactory.create_quantity("bulk_modulus", 100)
        self.assertIsInstance(q, NumQuantity)
        self.assertIsInstance(q.value, ureg.Quantity)
        self.assertEqual(q.value.magnitude, 100)
        self.assertEqual(q.value.units.format_babel(), 'gigapascal')
        # From custom symbol with constraint
        with self.assertRaises(SymbolConstraintError):
            QuantityFactory.create_quantity(self.constraint_symbol, -500)
        # From canonical symbol with constraint
        with self.assertRaises(SymbolConstraintError):
            QuantityFactory.create_quantity("bulk_modulus", -500)

        # Test np.array value with custom symbol
        value_list_array = [[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]]
        value_np_array = np.array(value_list_array)
        q = QuantityFactory.create_quantity(self.custom_symbol, value_np_array)
        self.assertIsInstance(q, NumQuantity)
        self.assertIsInstance(q.value, ureg.Quantity)
        self.assertTrue(np.allclose(q.value.magnitude, value_np_array))
        self.assertIsInstance(q.value.magnitude, np.ndarray)
        self.assertEqual(q.value.units.format_babel(), 'dimensionless')

        # Test list coercion
        q = QuantityFactory.create_quantity(self.custom_symbol, value_list_array)
        self.assertIsInstance(q, NumQuantity)
        self.assertTrue(np.allclose(q.value.magnitude, value_list_array))
        self.assertIsInstance(q.value.magnitude, np.ndarray)
        self.assertEqual(q.value.units.format_babel(), 'dimensionless')

        # From custom numerical symbol with uncertainty
        q = QuantityFactory.create_quantity(self.custom_symbol, 5.0, uncertainty=0.1)
        self.assertIsInstance(q, NumQuantity)
        self.assertIsInstance(q.value, ureg.Quantity)
        self.assertIsInstance(q.uncertainty, ureg.Quantity)
        self.assertAlmostEqual(q.value.magnitude, 5.0)
        self.assertAlmostEqual(q.uncertainty.magnitude, 0.1)
        self.assertEqual(q.value.units.format_babel(), 'dimensionless')
        self.assertEqual(q.uncertainty.units.format_babel(), 'dimensionless')

        # From canonical numerical symbol with uncertainty
        q = QuantityFactory.create_quantity("bulk_modulus", 100, uncertainty=1)
        self.assertIsInstance(q, NumQuantity)
        self.assertIsInstance(q.value, ureg.Quantity)
        self.assertIsInstance(q.uncertainty, ureg.Quantity)
        self.assertEqual(q.value.magnitude, 100)
        self.assertEqual(q.uncertainty.magnitude, 1)
        self.assertEqual(q.value.units.format_babel(), 'gigapascal')
        self.assertEqual(q.uncertainty.units.format_babel(), 'gigapascal')

        # Test np.array value with custom symbol with uncertainty
        uncertainty_list_array = [[vv*0.05 for vv in v] for v in value_list_array]
        value_np_array = np.array(value_list_array)
        uncertainty_np_array = np.array(uncertainty_list_array)
        q = QuantityFactory.create_quantity(self.custom_symbol, value_np_array,
                                            uncertainty=uncertainty_np_array)
        self.assertIsInstance(q, NumQuantity)
        self.assertIsInstance(q.value, ureg.Quantity)
        self.assertIsInstance(q.uncertainty, ureg.Quantity)
        self.assertIsInstance(q.value.magnitude, np.ndarray)
        self.assertIsInstance(q.uncertainty.magnitude, np.ndarray)
        self.assertTrue(np.allclose(q.value.magnitude, value_np_array))
        self.assertTrue(np.allclose(q.uncertainty.magnitude, uncertainty_np_array))
        self.assertEqual(q.value.units.format_babel(), 'dimensionless')
        self.assertEqual(q.uncertainty.units.format_babel(), 'dimensionless')

        # Test uncertainty list coercion
        q = QuantityFactory.create_quantity(self.custom_symbol, value_list_array,
                                            uncertainty=uncertainty_list_array)
        self.assertIsInstance(q, NumQuantity)
        self.assertIsInstance(q.value, ureg.Quantity)
        self.assertIsInstance(q.uncertainty, ureg.Quantity)
        self.assertIsInstance(q.value.magnitude, np.ndarray)
        self.assertIsInstance(q.uncertainty.magnitude, np.ndarray)
        self.assertTrue(np.allclose(q.value.magnitude, value_list_array))
        self.assertTrue(np.allclose(q.uncertainty.magnitude, uncertainty_np_array))
        self.assertEqual(q.value.units.format_babel(), 'dimensionless')
        self.assertEqual(q.uncertainty.units.format_babel(), 'dimensionless')

        # Test uncertainty NumQuantity coercion with unit conversion
        value_symbol = Symbol('E', units='joule')
        uncertainty_symbol = Symbol('u', units='calorie')
        incompatible_uncertainty_symbol = Symbol('u', units='meter')

        u = NumQuantity(uncertainty_symbol, 0.1)
        self.assertIsInstance(u, NumQuantity)
        self.assertIsInstance(u.value, ureg.Quantity)
        self.assertAlmostEqual(u.value.magnitude, 0.1)
        self.assertEqual(u.value.units.format_babel(), 'calorie')

        q = QuantityFactory.create_quantity(value_symbol, 126.1, uncertainty=u)
        self.assertIsInstance(q, NumQuantity)
        self.assertIsInstance(q.value, ureg.Quantity)
        self.assertIsInstance(q.uncertainty, ureg.Quantity)
        self.assertAlmostEqual(q.value.magnitude, 126.1)
        self.assertEqual(q.value.units.format_babel(), 'joule')
        self.assertAlmostEqual(q.uncertainty.magnitude, 0.4184)
        self.assertEqual(q.uncertainty.units.format_babel(), 'joule')

        u = NumQuantity(incompatible_uncertainty_symbol, 0.1)
        self.assertIsInstance(u, NumQuantity)
        self.assertIsInstance(u.value, ureg.Quantity)
        self.assertAlmostEqual(u.value.magnitude, 0.1)
        self.assertEqual(u.value.units.format_babel(), 'meter')
        with self.assertRaises(ValueError):
            QuantityFactory.create_quantity(value_symbol, 126.1, uncertainty=u)

        # Test uncertainty with bad type for uncertainty
        with self.assertRaises(TypeError):
            QuantityFactory.create_quantity(value_symbol, 126.1, uncertainty='test')

        # From custom object symbol
        q = QuantityFactory.create_quantity(self.custom_object_symbol, 'test')
        self.assertIsInstance(q, ObjQuantity)
        self.assertIsInstance(q.value, str)
        self.assertEqual(q.value, 'test')

        # From canonical object symbol
        q = QuantityFactory.create_quantity("is_metallic", False)
        self.assertIsInstance(q, ObjQuantity)
        self.assertIsInstance(q.value, bool)
        self.assertEqual(q.value, False)

        # Test failure with invalid symbol name
        with self.assertRaises(ValueError):
            QuantityFactory.create_quantity("my_invalid_symbol_name", 1)

        # Test failure with incorrect type as symbol
        with self.assertRaises(TypeError):
            QuantityFactory.create_quantity(self.custom_symbol.as_dict(), 100)

        # Test failure on instantiating NumQuantity with non-numeric types
        with self.assertRaises(TypeError):
            value = 'test_string'
            q = NumQuantity(self.custom_symbol, value)
        value_string_list_array = [['this', 'is'],
                                   ['a', 'test']]
        value_string_np_array = np.array(value_string_list_array)
        with self.assertRaises(TypeError):
            NumQuantity(self.custom_symbol, value_string_list_array)
        with self.assertRaises(TypeError):
            NumQuantity(self.custom_symbol, value_string_np_array)

        # Test failure initializing with NoneType objects
        with self.assertRaises(ValueError):
            QuantityFactory.create_quantity(self.custom_symbol, None)
        with self.assertRaises(ValueError):
            QuantityFactory.create_quantity(self.custom_object_symbol, None)
        with self.assertRaises(TypeError):
            NumQuantity(self.custom_symbol, None)
        with self.assertRaises(ValueError):
            ObjQuantity(self.custom_object_symbol, None)

        # Reject non-numerical quantities for non-object symbols
        with self.assertRaises(TypeError):
            QuantityFactory.create_quantity(self.custom_symbol, value_string_list_array)

        # Ensure warning is issued when assigning units or uncertainty to object-type symbol

        # Get logger where output is expected
        logger = logging.getLogger(QuantityFactory.__module__)

        # Test for warning with units
        with LogSniffer(logger) as ls:
            q = QuantityFactory.create_quantity(
                self.custom_object_symbol, 'test', units='dimensionless')

            log_output = ls.get_output(replace_newline='')
            expected_log_output = \
                "Cannot assign units to object-type symbol '{}'. " \
                "Ignoring units.".format(self.custom_object_symbol.name)

            self.assertEqual(log_output, expected_log_output)

        # Test for warning with uncertainty
        with LogSniffer(logger) as ls:
            q = QuantityFactory.create_quantity(self.custom_object_symbol, 'test', uncertainty='very uncertain')

            log_output = ls.get_output(replace_newline='')
            expected_log_output = \
                "Cannot assign uncertainty to object-type symbol '{}'. " \
                "Ignoring uncertainty.".format(self.custom_object_symbol.name)

            self.assertEqual(log_output, expected_log_output)

    def test_from_default(self):
        default = QuantityFactory.from_default('temperature')
        new_q = QuantityFactory.create_quantity('temperature', 300)
        # This test used to check for equality of the quantity objects,
        # but bc new definition of equality checks provenance, equality
        # between these objects fails (they originate from different models).
        # Now checking explicitly for symbol and value equality.
        self.assertEqual(default.symbol, new_q.symbol)
        self.assertEqual(default.value, new_q.value)
        default = QuantityFactory.from_default('relative_permeability')
        new_q = QuantityFactory.create_quantity("relative_permeability", 1)
        self.assertEqual(default.symbol, new_q.symbol)
        self.assertEqual(default.value, new_q.value)

        with self.assertRaises(ValueError):
            QuantityFactory.from_default('band_gap')

    def test_from_weighted_mean(self):
        qlist = [QuantityFactory.create_quantity(self.custom_symbol, val, tags=['testing'])
                 for val in np.arange(1, 2.01, 0.01)]
        qagg = NumQuantity.from_weighted_mean(qlist)
        self.assertAlmostEqual(qagg.magnitude, 1.5)
        self.assertAlmostEqual(qagg.uncertainty, 0.2915475947422652)
        self.assertListEqual(qagg.tags, ['testing'])

        qlist.append(QuantityFactory.create_quantity(Symbol('B', units='dimensionless'), 15))
        with self.assertRaises(ValueError):
            NumQuantity.from_weighted_mean(qlist)

        with self.assertRaises(ValueError):
            qlist = [QuantityFactory.create_quantity(self.custom_object_symbol, str(val))
                     for val in np.arange(1, 2.01, 0.01)]
            NumQuantity.from_weighted_mean(qlist)

    def test_is_cyclic(self):
        b = Symbol("B", units="dimensionless")
        q_lowest_layer = QuantityFactory.create_quantity(self.custom_symbol, 1)
        q_middle_layer = QuantityFactory.create_quantity(b, 2,
                                                         provenance=ProvenanceElement(model='A_to_B',
                                                                                      inputs=[q_lowest_layer]))
        q_highest_layer = QuantityFactory.create_quantity(self.custom_symbol, 1,
                                                          provenance=ProvenanceElement(model='B_to_A',
                                                                                       inputs=[q_middle_layer]))
        self.assertFalse(q_lowest_layer.is_cyclic())
        self.assertFalse(q_middle_layer.is_cyclic())
        self.assertTrue(q_highest_layer.is_cyclic())

        # A quantity should never not have a provenance assigned,
        # but that could be changed in the future or someone could
        # override it like below.
        q_no_provenance = QuantityFactory.create_quantity(self.custom_symbol, 1)
        q_no_provenance._provenance = None

        self.assertFalse(q_no_provenance.is_cyclic())

    def test_pretty_string(self):
        quantity = QuantityFactory.create_quantity('bulk_modulus', 100)
        self.assertEqual(quantity.pretty_string(sigfigs=3), "100 GPa")

        quantity = QuantityFactory.create_quantity('bulk_modulus', 100,
                                                   uncertainty=1.23456)
        self.assertEqual(quantity.pretty_string(sigfigs=3), "100\u00B11.235 GPa")

        quantity = QuantityFactory.create_quantity(self.custom_symbol, [1, 2, 3], 'dimensionless')
        self.assertEqual(quantity.pretty_string(), '[1 2 3]')

        quantity = QuantityFactory.create_quantity(self.custom_symbol, [1, 2, 3], 'atoms')
        self.assertEqual(quantity.pretty_string(), '[1 2 3] atom')

        quantity = QuantityFactory.create_quantity(self.custom_object_symbol, 'test')
        self.assertEqual(quantity.pretty_string(), 'test')

        quantity = QuantityFactory.create_quantity(self.custom_object_symbol, {'a': True})
        self.assertEqual(quantity.pretty_string(), "{'a': True}")

    def test_to(self):
        quantity = QuantityFactory.create_quantity('band_gap', 3.0, 'eV')
        new = quantity.to('joules')
        self.assertAlmostEqual(new.magnitude, 4.80652959e-19)
        self.assertEqual(new.units, 'joule')

        # Test with uncertainty
        quantity = QuantityFactory.create_quantity('band_gap', 3.0, 'eV',
                                                   uncertainty=0.1)
        new = quantity.to('joules')
        self.assertAlmostEqual(new.magnitude, 4.80652959e-19)
        self.assertEqual(new.units, 'joule')
        self.assertAlmostEqual(new.uncertainty.magnitude, 1.60217653e-20)
        self.assertEqual(new.uncertainty.units.format_babel(), 'joule')

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

        # Test truncation of long labels
        out = list(evaluated['sound_velocity_longitudinal'])[0]
        pgraph = out.get_provenance_graph()
        node_label = nx.get_node_attributes(pgraph, 'label')[out]
        self.assertEqual(node_label, 'sound_velocity_longitudinal')

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

    def test_type_coercion(self):
        # Numerical type coercion
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

        # Object type coercion for primitive type
        q = QuantityFactory.create_quantity("is_metallic", 0)
        self.assertIsInstance(q, ObjQuantity)
        self.assertIsInstance(q.value, bool)
        self.assertEqual(q.value, False)

        # For custom class
        # Test failure if module not imported
        s = Symbol('A', category='object',
                   object_type='propnet.core.tests.external_test_class.ACoercibleClass')
        with self.assertRaises(NameError):
            QuantityFactory.create_quantity(s, 55)

        # Test coercion after module imported
        from propnet.core.tests.external_test_class import ACoercibleClass, AnIncoercibleClass
        q = QuantityFactory.create_quantity(s, 55)
        self.assertIsInstance(q, ObjQuantity)
        self.assertIsInstance(q.value, ACoercibleClass)

        # Test coercion failure by failed typecasting
        s = Symbol('A', category='object',
                   object_type='propnet.core.tests.external_test_class.AnIncoercibleClass')
        with self.assertRaises(TypeError):
            QuantityFactory.create_quantity(s, 55)

        # Test lack of coercion when no type specified
        q = QuantityFactory.create_quantity(self.custom_object_symbol, AnIncoercibleClass(5, 6))
        self.assertIsInstance(q, ObjQuantity)
        self.assertIsInstance(q.value, AnIncoercibleClass)

        # Test coercion using QuantityFactory.to_quantity(). The difference between to_quantity()
        # and create_quantity() is that create_quantity() always creates a new object, whereas
        # to_quantity() will return the same object if it receives a BaseQuantity-derived object.
        # For non-BaseQuantity-derived objects, their functionality is exactly the same.

        # NumQuantity tests
        q_in = QuantityFactory.create_quantity(self.custom_symbol, 123, units='dimensionless')
        q_out = QuantityFactory.to_quantity(self.custom_symbol, q_in)
        self.assertIs(q_in, q_out)

        q_cq = QuantityFactory.create_quantity(self.custom_symbol, q_in, units='dimensionless')
        self.assertIsNot(q_in, q_cq)
        self.assertEqual(q_in, q_cq)

        q_tq = QuantityFactory.to_quantity(self.custom_symbol, 123, units='dimensionless')
        self.assertIsNot(q_in, q_tq)
        self.assertEqual(q_in, q_tq)

        # ObjQuantity tests
        q_in = QuantityFactory.create_quantity(self.custom_object_symbol, 'test')
        q_out = QuantityFactory.to_quantity(self.custom_object_symbol, q_in)
        self.assertIs(q_in, q_out)

        q_cq = QuantityFactory.create_quantity(self.custom_object_symbol, q_in)
        self.assertIsNot(q_in, q_cq)
        self.assertEqual(q_in, q_cq)

        q_tq = QuantityFactory.to_quantity(self.custom_object_symbol, 'test')
        self.assertIsNot(q_in, q_tq)
        self.assertEqual(q_in, q_tq)

    def test_as_dict_from_dict(self):
        q = QuantityFactory.create_quantity(self.custom_symbol, 5, tags='experimental', uncertainty=1)
        d = q.as_dict()

        self.assertDictEqual(d, {"@module": "propnet.core.quantity",
                                 "@class": "NumQuantity",
                                 "value": 5,
                                 "units": "dimensionless",
                                 "provenance": q.provenance.as_dict(),
                                 "symbol_type": self.custom_symbol.as_dict(),
                                 "tags": ['experimental'],
                                 "uncertainty": (1, ()),
                                 "internal_id": q._internal_id
                                 })

        q_from = QuantityFactory.from_dict(d)

        self.assertIsInstance(q_from, NumQuantity)
        self.assertEqual(q_from.symbol, q.symbol)
        self.assertEqual(q_from.value, q.value)
        self.assertEqual(q_from.units, q.units)
        self.assertEqual(q_from.tags, q.tags)
        self.assertEqual(q_from.uncertainty, q.uncertainty)
        self.assertEqual(q_from.provenance, q.provenance)
        self.assertEqual(q_from._internal_id, q._internal_id)

        q_from = NumQuantity.from_dict(d)

        self.assertIsInstance(q_from, NumQuantity)
        self.assertEqual(q_from.symbol, q.symbol)
        self.assertEqual(q_from.value, q.value)
        self.assertEqual(q_from.units, q.units)
        self.assertEqual(q_from.tags, q.tags)
        self.assertEqual(q_from.uncertainty, q.uncertainty)
        self.assertEqual(q_from.provenance, q.provenance)
        self.assertEqual(q_from._internal_id, q._internal_id)

        q = QuantityFactory.create_quantity(Registry("symbols")['debye_temperature'],
                                            500, tags='experimental', uncertainty=10)
        d = q.as_dict()

        self.assertDictEqual(d, {"@module": "propnet.core.quantity",
                                 "@class": "NumQuantity",
                                 "value": 500,
                                 "units": "kelvin",
                                 "provenance": q.provenance.as_dict(),
                                 "symbol_type": "debye_temperature",
                                 "tags": ['experimental'],
                                 "uncertainty": (10, (('kelvin', 1.0),)),
                                 "internal_id": q._internal_id
                                 })

        q_from = QuantityFactory.from_dict(d)

        self.assertIsInstance(q_from, NumQuantity)
        self.assertEqual(q_from.symbol, q.symbol)
        self.assertEqual(q_from.value, q.value)
        self.assertEqual(q_from.units, q.units)
        self.assertEqual(q_from.tags, q.tags)
        self.assertEqual(q_from.uncertainty, q.uncertainty)
        self.assertEqual(q_from.provenance, q.provenance)
        self.assertEqual(q_from._internal_id, q._internal_id)

        q_from = NumQuantity.from_dict(d)

        self.assertIsInstance(q_from, NumQuantity)
        self.assertEqual(q_from.symbol, q.symbol)
        self.assertEqual(q_from.value, q.value)
        self.assertEqual(q_from.units, q.units)
        self.assertEqual(q_from.tags, q.tags)
        self.assertEqual(q_from.uncertainty, q.uncertainty)
        self.assertEqual(q_from.provenance, q.provenance)
        self.assertEqual(q_from._internal_id, q._internal_id)

        # Test ObjQuantity
        q = QuantityFactory.create_quantity(self.custom_object_symbol, 'test', tags='test_tag')
        d = q.as_dict()

        self.assertDictEqual(d, {"@module": "propnet.core.quantity",
                                 "@class": "ObjQuantity",
                                 "value": 'test',
                                 "provenance": q.provenance.as_dict(),
                                 "symbol_type": self.custom_object_symbol.as_dict(),
                                 "tags": ['test_tag'],
                                 "internal_id": q._internal_id
                                 })

        q_from = QuantityFactory.from_dict(d)

        self.assertIsInstance(q_from, ObjQuantity)
        self.assertEqual(q_from.symbol, q.symbol)
        self.assertEqual(q_from.value, q.value)
        self.assertEqual(q_from.tags, q.tags)
        self.assertEqual(q_from.provenance, q.provenance)
        self.assertEqual(q_from._internal_id, q._internal_id)

        q_from = ObjQuantity.from_dict(d)

        self.assertIsInstance(q_from, ObjQuantity)
        self.assertEqual(q_from.symbol, q.symbol)
        self.assertEqual(q_from.value, q.value)
        self.assertEqual(q_from.tags, q.tags)
        self.assertEqual(q_from.provenance, q.provenance)
        self.assertEqual(q_from._internal_id, q._internal_id)

        q = QuantityFactory.create_quantity(Registry("symbols")['is_metallic'],
                                            False, tags='dft')
        d = q.as_dict()

        self.assertDictEqual(d, {"@module": "propnet.core.quantity",
                                 "@class": "ObjQuantity",
                                 "value": False,
                                 "provenance": q.provenance.as_dict(),
                                 "symbol_type": "is_metallic",
                                 "tags": ['dft'],
                                 "internal_id": q._internal_id
                                 })

        q_from = QuantityFactory.from_dict(d)

        self.assertIsInstance(q_from, ObjQuantity)
        self.assertEqual(q_from.symbol, q.symbol)
        self.assertEqual(q_from.value, q.value)
        self.assertEqual(q_from.tags, q.tags)
        self.assertEqual(q_from.provenance, q.provenance)
        self.assertEqual(q_from._internal_id, q._internal_id)

        q_from = ObjQuantity.from_dict(d)

        self.assertIsInstance(q_from, ObjQuantity)
        self.assertEqual(q_from.symbol, q.symbol)
        self.assertEqual(q_from.value, q.value)
        self.assertEqual(q_from.tags, q.tags)
        self.assertEqual(q_from.provenance, q.provenance)
        self.assertEqual(q_from._internal_id, q._internal_id)

        # Test failure of QuantityFactory.from_dict() with non BaseQuantity objects
        d_provenance = q.provenance.as_dict()
        with self.assertRaises(ValueError):
            QuantityFactory.from_dict(d_provenance)

    def test_equality(self):
        q1 = QuantityFactory.create_quantity(self.custom_symbol, 5, tags='experimental', uncertainty=1)
        q1_copy = copy.deepcopy(q1)
        q2 = QuantityFactory.create_quantity(self.custom_symbol, 6, tags='experimental', uncertainty=2)
        q3 = QuantityFactory.create_quantity(self.custom_symbol, 6, tags='experimental')
        q4 = QuantityFactory.create_quantity(Symbol('test_symbol', units='dimensionless'), 5)

        q1_different_provenance = QuantityFactory.create_quantity(
            self.custom_symbol, 5, tags='experimental', uncertainty=1,
            provenance=ProvenanceElement(model='my_model', inputs=[q2]))

        # Tests using __eq__() method
        self.assertEqual(q1, q1_copy)
        self.assertEqual(q1.symbol, q1_copy.symbol)
        self.assertEqual(q1.value, q1_copy.value)
        self.assertEqual(q1.units, q1_copy.units)
        self.assertEqual(q1.tags, q1_copy.tags)
        self.assertEqual(q1.uncertainty, q1_copy.uncertainty)
        self.assertEqual(q1.provenance, q1_copy.provenance)

        # Negative __eq__() tests
        self.assertNotEqual(q1, q2)
        self.assertNotEqual(q2, q3)
        self.assertNotEqual(q1, q1_different_provenance)

        # Tests using has_eq_value_to() which ignores provenance
        # and only looks at the value and units
        self.assertTrue(q1.has_eq_value_to(q1_copy))
        self.assertTrue(q1.has_eq_value_to(q1_different_provenance))
        self.assertTrue(q1.has_eq_value_to(q4))
        self.assertFalse(q1.has_eq_value_to(q2))
        self.assertTrue(q2.has_eq_value_to(q3))
        with self.assertRaises(TypeError):
            q1.has_eq_value_to(5)

        # Quantity creation to test value equality in different units using values_close_in_units().
        # eV are small energy units and joule are large energy units.
        # In principle, comparing units in eV vs. joules vs. attempting to discover the appropriate
        # unit may yield different results. This is an attempt to robustly equate values which are floats.

        q_ev = QuantityFactory.create_quantity('band_gap', 1.0, units='eV')
        # Should be equal to q_ev within tolerance
        q_ev_slightly_bigger = QuantityFactory.create_quantity('band_gap', 1. + 1e-8, units='eV')
        # Should not be equal to q_ev within tolerance
        q_ev_too_big = QuantityFactory.create_quantity('band_gap', 1. + 1e-4, units='eV')

        # Function requires pint quantities as inputs
        with self.assertRaises(TypeError):
            NumQuantity.values_close_in_units(q_ev, q_ev_slightly_bigger)

        self.assertTrue(NumQuantity.values_close_in_units(q_ev.value, q_ev_slightly_bigger.value))
        self.assertFalse(NumQuantity.values_close_in_units(q_ev.value, q_ev_too_big.value))

        q_ev_zero = QuantityFactory.create_quantity('band_gap', 0., units='eV')
        q_joule_zero = q_ev_zero.to('joule')
        q_nev_zero = q_ev_zero.to('nanoelectron_volt')
        # When compared to 0 eV in eV, this should be considered close to zero.
        # When compared to 0 neV in neV, will not be close to zero
        q_ev_close_to_zero = QuantityFactory.create_quantity('band_gap', 1e-8, units='eV')
        # When compared to 0 J in J, this should be considered close to zero, but not when compared in eV
        q_ev_close_to_zero_in_joules = QuantityFactory.create_quantity('band_gap', 0.1, units='eV')

        # Does not do unit comparison because they should both have value == zero is True
        # This assumes there were no floating point rounding errors
        self.assertEqual(q_ev_zero.magnitude, 0)
        self.assertEqual(q_joule_zero.magnitude, 0)
        self.assertTrue(NumQuantity.values_close_in_units(
            q_ev_zero.value, q_joule_zero.value))
        self.assertTrue(NumQuantity.values_close_in_units(
            q_joule_zero.value, q_ev_zero.value))

        # Compares in eV implicitly (from unit of quantity that is 0)
        self.assertTrue(NumQuantity.values_close_in_units(
            q_ev_zero.value, q_ev_close_to_zero.value))
        self.assertTrue(NumQuantity.values_close_in_units(
            q_ev_close_to_zero.value, q_ev_zero.value))
        # Compares in joules implicitly (from unit of quantity that is 0)
        self.assertTrue(NumQuantity.values_close_in_units(
            q_ev_close_to_zero.value, q_joule_zero.value))
        self.assertTrue(NumQuantity.values_close_in_units(
            q_joule_zero.value, q_ev_close_to_zero.value))
        # Compares in neV implicitly (from unit of quantity that is 0)
        self.assertFalse(NumQuantity.values_close_in_units(
            q_nev_zero.value, q_ev_close_to_zero.value))
        self.assertFalse(NumQuantity.values_close_in_units(
            q_ev_close_to_zero.value, q_nev_zero.value))

        # Compares in joules explicitly
        self.assertTrue(NumQuantity.values_close_in_units(
            q_ev_close_to_zero_in_joules.value, q_ev_zero.value,
            units_for_comparison='joule'))
        self.assertTrue(NumQuantity.values_close_in_units(
            q_ev_zero.value, q_ev_close_to_zero_in_joules.value,
            units_for_comparison='joule'))
        # Compares in eV explicitly
        self.assertFalse(NumQuantity.values_close_in_units(
            q_ev_close_to_zero_in_joules.value, q_ev_zero.value,
            units_for_comparison='eV'))
        self.assertFalse(NumQuantity.values_close_in_units(
            q_ev_zero.value, q_ev_close_to_zero_in_joules.value,
            units_for_comparison='eV'))

        # Compares in eV implicitly (from unit of quantity that is 0)
        self.assertFalse(NumQuantity.values_close_in_units(
            q_ev_close_to_zero_in_joules.value, q_ev_zero.value))
        self.assertFalse(NumQuantity.values_close_in_units(
            q_ev_zero.value, q_ev_close_to_zero_in_joules.value))
        # Compares in joules implicitly (from unit of quantity that is 0)
        self.assertTrue(NumQuantity.values_close_in_units(
            q_ev_close_to_zero_in_joules.value, q_joule_zero.value))
        self.assertTrue(NumQuantity.values_close_in_units(
            q_joule_zero.value, q_ev_close_to_zero_in_joules.value))

        # These two numbers are small in eV and should both be considered close to zero when evaluated in eV.
        # If units for comparison are not specified, these values should be compared explicitly and not be close
        # in value (would be evaluated in femto-eV)
        q_ev_small_number_1 = QuantityFactory.create_quantity('band_gap', 1e-12, units='eV')
        q_ev_small_number_2 = QuantityFactory.create_quantity('band_gap', 1e-15, units='eV')

        self.assertFalse(NumQuantity.values_close_in_units(
            q_ev_small_number_1.value, q_ev_small_number_2.value))
        self.assertFalse(NumQuantity.values_close_in_units(
            q_ev_small_number_2.value, q_ev_small_number_1.value))
        self.assertFalse(NumQuantity.values_close_in_units(
            q_ev_small_number_1.value, q_ev_small_number_2.value,
            units_for_comparison='femtoelectron_volt'))
        self.assertFalse(NumQuantity.values_close_in_units(
            q_ev_small_number_2.value, q_ev_small_number_1.value,
            units_for_comparison='femtoelectron_volt'))
        self.assertTrue(NumQuantity.values_close_in_units(
            q_ev_small_number_1.value, q_ev_small_number_2.value,
            units_for_comparison='eV'))
        self.assertTrue(NumQuantity.values_close_in_units(
            q_ev_small_number_2.value, q_ev_small_number_1.value,
            units_for_comparison='eV'))

        # Compare values with unit with comparable magnitude (hartree and eV are both molecular in magnitude)
        q_hartree = q_ev.to('hartree')
        self.assertTrue(NumQuantity.values_close_in_units(q_ev.value, q_hartree.value))
        self.assertTrue(NumQuantity.values_close_in_units(q_hartree.value, q_ev.value))

        # Convert above values to joules to try the same comparisons
        q_joule = q_ev.to('joule')
        q_joule_slightly_bigger = q_ev_slightly_bigger.to('joule')
        q_joule_too_big = q_ev_too_big.to('joule')
        q_joule_close_to_zero = q_ev_close_to_zero.to('joule')
        q_joule_close_to_zero_in_joules = q_ev_close_to_zero_in_joules.to('joule')

        # If values are same, but in different units, coerce to units that make value between 1 and 1000
        # and choose unit that is smaller between the two
        # Evaluates in zeptojoules
        self.assertTrue(NumQuantity.values_close_in_units(
            q_ev.value, q_joule.value))
        self.assertTrue(NumQuantity.values_close_in_units(
            q_joule.value, q_ev.value))
        self.assertTrue(NumQuantity.values_close_in_units(
            q_joule.value, q_joule_slightly_bigger.value))
        self.assertTrue(NumQuantity.values_close_in_units(
            q_joule_slightly_bigger.value, q_joule.value))
        self.assertFalse(NumQuantity.values_close_in_units(
            q_joule.value, q_joule_too_big.value))
        self.assertFalse(NumQuantity.values_close_in_units(
            q_joule_too_big.value, q_joule.value))
        self.assertTrue(NumQuantity.values_close_in_units(
            q_joule_zero.value, q_joule_close_to_zero.value))
        self.assertTrue(NumQuantity.values_close_in_units(
            q_joule_close_to_zero.value, q_joule_zero.value))

        # Test units for comparison defaults to eV based on symbol unit
        self.assertEqual(q_ev_zero.symbol.units.format_babel(), 'electron_volt')
        self.assertEqual(q_ev_close_to_zero, q_ev_zero)
        self.assertEqual(q_ev_zero, q_ev_close_to_zero)
        self.assertEqual(q_joule_close_to_zero, q_joule_zero)
        self.assertEqual(q_joule_zero, q_joule_close_to_zero)
        self.assertNotEqual(q_joule_close_to_zero_in_joules, q_joule_zero)
        self.assertNotEqual(q_joule_zero, q_joule_close_to_zero_in_joules)
        self.assertEqual(q_ev_small_number_1, q_ev_small_number_2)
        self.assertEqual(q_ev_small_number_2, q_ev_small_number_1)

        # Test with ndarrays
        array_sym = Symbol("M", units='eV', shape=[4])
        q_ev_array = QuantityFactory.create_quantity(
            array_sym, [1., 2., 3., 1e-8], units='eV')
        q_joule_array = q_ev_array.to('joule')

        self.assertEqual(q_ev_array, q_joule_array)
        self.assertTrue(NumQuantity.values_close_in_units(
            q_ev_array.value, q_joule_array.value))
        self.assertTrue(NumQuantity.values_close_in_units(
            q_joule_array.value, q_ev_array.value))

        # Test with objects (much simpler...)
        q = QuantityFactory.create_quantity(self.custom_object_symbol, 'test')
        q_duplicate = QuantityFactory.create_quantity(self.custom_object_symbol, 'test')
        q_copy = copy.deepcopy(q)
        q_tags = QuantityFactory.create_quantity(self.custom_object_symbol, 'test', tags='custom')
        q_provenance = QuantityFactory.create_quantity(self.custom_object_symbol, 'test',
                                                       tags='custom',
                                                       provenance=ProvenanceElement(model='my_model',
                                                                                    inputs=[q]))
        self.assertEqual(q, q_duplicate)
        self.assertEqual(q, q_copy)
        self.assertNotEqual(q, q_tags)
        self.assertNotEqual(q, q_provenance)

        # Cannot compare ObjQuantity to NumQuantity
        with self.assertRaises(TypeError):
            q.has_eq_value_to(q_ev)

        fields = list(q1.__dict__.keys())

        # This is to check to see if we modified the fields in the object, in case we need to add
        # to our equality statement
        self.assertListEqual(fields, ['_value', '_symbol_type',
                                      '_tags', '_provenance',
                                      '_internal_id', '_uncertainty'])

    def test_hash(self):
        q = QuantityFactory.create_quantity(self.custom_symbol, 1, 'dimensionless')
        q_duplicate = QuantityFactory.create_quantity(self.custom_symbol, 1, 'dimensionless')
        q_copy = copy.deepcopy(q)
        q_diff_value_same_provenance = QuantityFactory.create_quantity(self.custom_symbol, 2, 'dimensionless')
        q_tags = QuantityFactory.create_quantity(self.custom_symbol, 1, 'dimensionless',
                                                 tags='experimental')
        q_provenance = QuantityFactory.create_quantity(self.custom_symbol, 1, 'dimensionless',
                                                       provenance=ProvenanceElement(
                                                           model='my_model',
                                                           inputs=[q, q_tags]
                                                       ))

        s = {q, q_duplicate}
        self.assertTrue(len(s), 1)
        s.add(q_copy)
        self.assertTrue(len(s), 1)
        s.add(q_diff_value_same_provenance)
        # hash() is same but __eq__() is different
        self.assertTrue(hash(q) == hash(q_diff_value_same_provenance))
        self.assertTrue(len(s), 2)
        s.add(q_tags)
        self.assertTrue(len(s), 3)
        s.add(q_provenance)
        self.assertTrue(len(s), 4)

    def test_bool(self):
        # Currently one cannot assign None to the value of a quantity, but that
        # may change in the future.
        # TODO: Consider whether we need to be able to assign None to values
        q = QuantityFactory.create_quantity(self.custom_symbol, 1, 'dimensionless')
        self.assertTrue(q)
        q._value = None
        self.assertFalse(q)

        q = QuantityFactory.create_quantity(self.custom_object_symbol, 'test')
        self.assertTrue(q)
        q._value = None
        self.assertFalse(q)
        q._value = ''
        self.assertFalse(q)

    def test_str_and_repr(self):
        q = QuantityFactory.create_quantity(self.custom_symbol, 1, 'dimensionless', tags='custom')
        self.assertEqual(str(q), "<A, 1 dimensionless, ['custom']>")
        self.assertEqual(repr(q), "<A, 1 dimensionless, ['custom']>")


if __name__ == "__main__":
    unittest.main()
