import unittest
from propnet import ureg
from propnet.core.materials import Material
from propnet.core.quantity import Quantity
from propnet.core.graph import Graph
from propnet.symbols import DEFAULT_SYMBOLS

class MaterialTest(unittest.TestCase):

    def setUp(self):
        # Create some test properties and a few base objects
        self.q1 = Quantity(DEFAULT_SYMBOLS['bulk_modulus'],
                           ureg.Quantity.from_tuple([200, [['gigapascals', 1]]]))
        self.q2 = Quantity(DEFAULT_SYMBOLS['shear_modulus'],
                           ureg.Quantity.from_tuple([100, [['gigapascals', 1]]]))
        self.q3 = Quantity(DEFAULT_SYMBOLS['bulk_modulus'],
                           ureg.Quantity.from_tuple([300, [['gigapascals', 1]]]))
        self.material = Material()
        self.graph = Graph()

    def test_material_setup(self):
        self.assertTrue(self.material.parent is None
                        and len(self.material._symbol_to_quantity) == 0
                        and self.material.uuid is not None,
                        "Material not initialized properly.")
        self.graph.add_material(self.material)
        self.assertTrue(self.material.parent is self.graph,
                        "Material not added to graph properly.")
        self.assertEqual(len(self.material._symbol_to_quantity), 0,
                         "Material not added to graph properly.")

    def test_material_add_quantity(self):
        self.material.add_quantity(self.q1)
        self.assertTrue(self.material.parent is None,
                        "Material did not add the quantity.")
        self.assertEqual(len(self.material._symbol_to_quantity), 1,
                         "Material did not add the quantity.")
        self.graph.add_material(self.material)
        self.assertEqual(len(self.graph._symbol_to_quantity), 1,
                        "Material did not add quantity to the graph: q1.")
        self.material.add_quantity(self.q2)
        self.assertEqual(len(self.material._symbol_to_quantity), 2,
                         "Material did not add quantity to itself.")
        self.assertEqual(len(self.graph._symbol_to_quantity), 2,
                         "Material did not add quantity to the graph: q2.")

    def test_material_remove_quantity(self):
        self.material.add_quantity(self.q1)
        self.material.add_quantity(self.q2)
        self.material.remove_quantity(self.q1)
        self.assertEqual(
            len(self.material._symbol_to_quantity[DEFAULT_SYMBOLS['shear_modulus']]),
            1, "Material did not remove the correct quantity.")
        self.graph.add_material(self.material)
        self.material.remove_quantity(self.q2)
        self.assertEqual(
            len(self.graph._symbol_to_quantity[DEFAULT_SYMBOLS['shear_modulus']]),
            0, "Material did not remove the quantity correctly.")
        self.assertEqual(
            len(self.material._symbol_to_quantity[DEFAULT_SYMBOLS['shear_modulus']]),
            0, "Material did not remove the quantity correctly.")

    def test_material_remove_symbol(self):
        self.material.add_quantity(self.q1)
        self.material.add_quantity(self.q2)
        self.material.add_quantity(self.q3)
        self.material.remove_symbol(DEFAULT_SYMBOLS['bulk_modulus'])
        self.assertTrue(
            DEFAULT_SYMBOLS['shear_modulus'] in self.material._symbol_to_quantity.keys(),
            "Material did not remove Symbol correctly.")
        self.assertTrue(
            self.q2 in self.material._symbol_to_quantity[DEFAULT_SYMBOLS['shear_modulus']],
            "Material did not remove Symbol correctly.")
        self.assertEqual(len(self.material._symbol_to_quantity), 1,
                         "Material did not remove Symbol correctly.")

    def test_get_symbols(self):
        self.material.add_quantity(self.q1)
        self.material.add_quantity(self.q2)
        self.material.add_quantity(self.q3)
        out = self.material.get_symbols()
        self.assertEqual(
            len(out), 2, "Material did not get Symbol Types correctly.")
        self.assertTrue(DEFAULT_SYMBOLS['bulk_modulus'] in out,
                        "Material did not get Symbol Types correctly.")
        self.assertTrue(DEFAULT_SYMBOLS['shear_modulus'] in out,
                        "Material did not get Symbol Types correctly.")

    def test_get_quantities(self):
        self.material.add_quantity(self.q1)
        self.material.add_quantity(self.q2)
        self.material.add_quantity(self.q3)
        out = self.material.get_quantities()
        self.assertTrue(all([q in out for q in [self.q1, self.q2, self.q3]]),
                        "Material did not get Quantity objects correctly.")

    def test_get_unique_quantities(self):
        self.material.add_quantity(self.q1)
        self.material.add_quantity(self.q2)
        m2 = Material()
        m2.add_quantity(self.q3)
        g = Graph(materials=[self.material, m2])
        g.evaluate()
        out1 = [x for x in self.material.get_quantities()
                if x.symbol == DEFAULT_SYMBOLS['pugh_ratio']]
        out2 = [x for x in self.material.get_unique_quantities()
                if x.symbol == DEFAULT_SYMBOLS['pugh_ratio']]
        self.assertTrue(len(out1) > len(out2) and
                        all([m2 not in x._material for x in out2]),
                        "Unique quantities did not correctly filter.")

    def test_get_aggregated_quantities(self):
        self.material.add_quantity(self.q1)
        self.material.add_quantity(self.q2)
        self.material.add_quantity(self.q3)
        agg = self.material.get_aggregated_quantities()
        # TODO: add a meaningful test here

    def test_evaluate(self):
        self.material.add_quantity(self.q1)
        self.material.add_quantity(self.q2)
        self.material.evaluate()
        self.assertGreater(len(self.material._symbol_to_quantity), 2,
                           "No new Symbols were evaluated.")
