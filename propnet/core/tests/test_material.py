import unittest
from propnet import ureg
from propnet.core.materials import Material
from propnet.core.quantity import create_quantity
from propnet.core.graph import Graph
from propnet.symbols import DEFAULT_SYMBOLS

class MaterialTest(unittest.TestCase):

    def setUp(self):
        # Create some test properties and a few base objects
        self.q1 = create_quantity(DEFAULT_SYMBOLS['bulk_modulus'],
                                  ureg.Quantity.from_tuple([200, [['gigapascals', 1]]]))
        self.q2 = create_quantity(DEFAULT_SYMBOLS['shear_modulus'],
                                  ureg.Quantity.from_tuple([100, [['gigapascals', 1]]]))
        self.q3 = create_quantity(DEFAULT_SYMBOLS['bulk_modulus'],
                                  ureg.Quantity.from_tuple([300, [['gigapascals', 1]]]))
        self.material = Material()
        self.graph = Graph()

    def test_material_setup(self):
        self.assertTrue(len(self.material._symbol_to_quantity) == 0,
                        "Material not initialized properly.")

    def test_material_add_quantity(self):
        self.material.add_quantity(self.q1)
        self.assertEqual(len(self.material._symbol_to_quantity), 1,
                         "Material did not add the quantity.")
        self.material.add_quantity(self.q2)
        self.assertEqual(len(self.material._symbol_to_quantity), 2,
                         "Material did not add quantity to itself.")

    def test_material_remove_quantity(self):
        self.material.add_quantity(self.q1)
        self.material.add_quantity(self.q2)
        self.material.remove_quantity(self.q1)
        self.assertEqual(
            len(self.material._symbol_to_quantity[DEFAULT_SYMBOLS['shear_modulus']]),
            1, "Material did not remove the correct quantity.")
        self.material.remove_quantity(self.q2)
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

    def test_get_aggregated_quantities(self):
        self.material.add_quantity(self.q1)
        self.material.add_quantity(self.q2)
        self.material.add_quantity(self.q3)
        agg = self.material.get_aggregated_quantities()
        # TODO: add a meaningful test here

    def test_add_default_quantities(self):
        material = Material(add_default_quantities=True)
        self.assertEqual(list(material['temperature'])[0],
                         create_quantity("temperature", 300))
        self.assertEqual(list(material['relative_permeability'])[0],
                         create_quantity("relative_permeability", 1))
