import unittest
import os
import pandas as pd

import numpy as np

from pymatgen.util.testing import PymatgenTest
from propnet.core.quantity import Quantity, get_sse, get_weight, \
    aggregate_quantities, fit_model_scores
from propnet.core.symbols import Symbol
from propnet.core.exceptions import SymbolConstraintError
from propnet.core.graph import Graph
from propnet.core.materials import Material
from propnet.core.provenance import ProvenanceElement
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


class FittingTests(unittest.TestCase):
    def setUp(self):
        path = os.path.join(TEST_DIR, "fitting_test_data.csv")
        test_data = pd.read_csv(path)
        graph = Graph()
        materials = [Material([Quantity("band_gap", bg)])
                     for bg in test_data['band_gap']]
        self.evaluated = [graph.evaluate(mat) for mat in materials]
        self.benchmarks = [{"refractive_index", n}
                           for n in test_data['refractive_index']]

    def test_get_sse(self):
        mats = [Material(Quantity("band_gap", n)) for n in range(1, 5)]
        benchmarks = [{"band_gap": 1.1*n} for n in range(1, 5)]
        err = get_sse(mats, benchmarks)
        test_val = sum([0.01*n**2 for n in range(1, 5)])
        self.assertEqual(err, test_val)
        # Big dataset
        err = get_sse(self.evaluated, self.benchmarks)
        self.assertEqual(err, 123.01)

    def test_get_weight(self):
        q1 = Quantity("band_gap", 3.2)
        wt = get_weight(q1)
        self.assertEqual(wt, 1)
        p2 = ProvenanceElement(model="model_2", inputs=[q1])
        q2 = Quantity("refractive_index", 4, provenance=p2)
        wt2 = get_weight(q2, {"model_2": 0.5})
        self.assertEqual(wt2, 0.5)
        p3 = ProvenanceElement(model="model_3", inputs=[q2])
        q3 = Quantity("bulk_modulus", 100, provenance=p3)
        wt3 = get_weight(q3, {"model_3": 0.25, "model_2": 0.5})
        self.assertEqual(wt3, 0.125)

    def test_fit_model_scores(self):
        scores = fit_model_scores(self.evaluated, self.benchmarks)
        self.assertEqual(scores['bug'])

    def dumb_test(self):
        from propnet.core.graph import Graph
        from propnet.ext.matproj import MPRester
        mpr = MPRester()
        g = Graph()
        mat = mpr.get_material_for_mpid("mp-66")
        mat.add_default_quantities()
        g.evaluate(mat)