import unittest
from propnet import ureg
from propnet.core.materials import Material
from propnet.core.quantity import Quantity
from propnet.core.graph import Graph
from propnet.symbols import DEFAULT_SYMBOLS
from propnet.models import DEFAULT_MODELS

class GraphTest(unittest.TestCase):

    def test_material_setup(self):
        m = Material()
        self.assertTrue(m.parent is None and len(m._symbol_to_quantity) == 0 and m.uuid is not None,
                        "Material not initialized properly.")
        g = Graph()
        g.add_material(m)
        self.assertTrue(m.parent is g and len(m._symbol_to_quantity) == 0,
                        "Material not added to graph properly.")

    def test_material_add_quantity(self):
        q1 = Quantity(DEFAULT_SYMBOLS['bulk_modulus'], ureg.Quantity.from_tuple([200, [['gigapascals', 1]]]))
        q2 = Quantity(DEFAULT_SYMBOLS['shear_modulus'], ureg.Quantity.from_tuple([100, [['gigapascals', 1]]]))
        m = Material()
        m.add_quantity(q1)
        self.assertTrue(m.parent is None and len(m._symbol_to_quantity) == 1,
                        "Material did not add the quantity.")
        g = Graph()
        g.add_material(m)
        self.assertTrue(len(g._symbol_to_quantity) == 1,
                        "Material did not add quantity to the graph: q1.")
        m.add_quantity(q2)
        self.assertTrue(len(m._symbol_to_quantity) == 2,
                        "Material did not add quantity to itself.")
        self.assertTrue(len(g._symbol_to_quantity) == 2,
                        "Material did not add quantity to the graph: q2.")

    def test_material_remove_quantity(self):
        q1 = Quantity(DEFAULT_SYMBOLS['bulk_modulus'], ureg.Quantity.from_tuple([200, [['gigapascals', 1]]]))
        q2 = Quantity(DEFAULT_SYMBOLS['shear_modulus'], ureg.Quantity.from_tuple([100, [['gigapascals', 1]]]))
        m = Material()
        m.add_quantity(q1)
        m.add_quantity(q2)
        m.remove_quantity(q1)
        self.assertTrue(len(m._symbol_to_quantity[DEFAULT_SYMBOLS['shear_modulus']]) == 1,
                       "Material did not remove the correct quantity.")
        g = Graph()
        g.add_material(m)
        m.remove_quantity(q2)
        self.assertTrue(len(g._symbol_to_quantity[DEFAULT_SYMBOLS['shear_modulus']]) == 0 and
                        len(m._symbol_to_quantity[DEFAULT_SYMBOLS['shear_modulus']]) == 0,
                        "Material did not remove the quantity correctly.")

    def test_material_remove_symbol(self):
        q1 = Quantity(DEFAULT_SYMBOLS['bulk_modulus'], ureg.Quantity.from_tuple([200, [['gigapascals', 1]]]))
        q2 = Quantity(DEFAULT_SYMBOLS['shear_modulus'], ureg.Quantity.from_tuple([100, [['gigapascals', 1]]]))
        q3 = Quantity(DEFAULT_SYMBOLS['bulk_modulus'], ureg.Quantity.from_tuple([300, [['gigapascals', 1]]]))
        m = Material()
        m.add_quantity(q1)
        m.add_quantity(q2)
        m.add_quantity(q3)
        m.remove_symbol(DEFAULT_SYMBOLS['bulk_modulus'])
        self.assertTrue(DEFAULT_SYMBOLS['shear_modulus'] in m._symbol_to_quantity.keys() and
                        q2 in m._symbol_to_quantity[DEFAULT_SYMBOLS['shear_modulus']] and
                        len(m._symbol_to_quantity) == 1,
                        "Material did not remove Symbol correctly.")

    def test_get_symbols(self):
        q1 = Quantity(DEFAULT_SYMBOLS['bulk_modulus'], ureg.Quantity.from_tuple([200, [['gigapascals', 1]]]))
        q2 = Quantity(DEFAULT_SYMBOLS['shear_modulus'], ureg.Quantity.from_tuple([100, [['gigapascals', 1]]]))
        q3 = Quantity(DEFAULT_SYMBOLS['bulk_modulus'], ureg.Quantity.from_tuple([300, [['gigapascals', 1]]]))
        m = Material()
        m.add_quantity(q1)
        m.add_quantity(q2)
        m.add_quantity(q3)
        out = m.get_symbols()
        self.assertTrue(len(out) == 2 and
                        DEFAULT_SYMBOLS['bulk_modulus'] in out and
                        DEFAULT_SYMBOLS['shear_modulus'] in out,
                        "Material did not get Symbol Types correctly.")

    def test_get_quantities(self):
        q1 = Quantity(DEFAULT_SYMBOLS['bulk_modulus'], ureg.Quantity.from_tuple([200, [['gigapascals', 1]]]))
        q2 = Quantity(DEFAULT_SYMBOLS['shear_modulus'], ureg.Quantity.from_tuple([100, [['gigapascals', 1]]]))
        q3 = Quantity(DEFAULT_SYMBOLS['bulk_modulus'], ureg.Quantity.from_tuple([300, [['gigapascals', 1]]]))
        m = Material()
        m.add_quantity(q1)
        m.add_quantity(q2)
        m.add_quantity(q3)
        out = m.get_quantities()
        self.assertTrue(q1 in out and q2 in out and q3 in out,
                        "Material did not get Quantity objects correctly.")

    def test_get_unique_quantities(self):
        q1 = Quantity(DEFAULT_SYMBOLS['bulk_modulus'], ureg.Quantity.from_tuple([200, [['gigapascals', 1]]]))
        q2 = Quantity(DEFAULT_SYMBOLS['shear_modulus'], ureg.Quantity.from_tuple([100, [['gigapascals', 1]]]))
        q3 = Quantity(DEFAULT_SYMBOLS['bulk_modulus'], ureg.Quantity.from_tuple([300, [['gigapascals', 1]]]))
        m1 = Material()
        m2 = Material()
        m1.add_quantity(q1)
        m1.add_quantity(q2)
        m2.add_quantity(q3)
        g = Graph(materials=[m1,m2])
        g.evaluate()
        out1 = [x for x in m1.get_quantities() if x.symbol == DEFAULT_SYMBOLS['pugh_ratio']]
        out2 = [x for x in m1.get_unique_quantities() if x.symbol == DEFAULT_SYMBOLS['pugh_ratio']]
        self.assertTrue(len(out1) > len(out2) and
                        all([m2 not in x._material for x in out2]),
                        "Unique quantities did not correctly filter.")

    def test_evaluate(self):
        q1 = Quantity(DEFAULT_SYMBOLS['bulk_modulus'], ureg.Quantity.from_tuple([200, [['gigapascals', 1]]]))
        q2 = Quantity(DEFAULT_SYMBOLS['shear_modulus'], ureg.Quantity.from_tuple([100, [['gigapascals', 1]]]))
        m1 = Material()
        m1.add_quantity(q1)
        m1.add_quantity(q2)
        m1.evaluate()
        self.assertTrue(len(m1._symbol_to_quantity) > 2,
                        "No new Symbols were evaluated.")
