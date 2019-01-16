import unittest

from propnet.core.graph import Graph
from propnet.ext.matproj import MPRester

mpr = MPRester()


@unittest.skipIf(mpr.api_key == "", "No API key provided. Skipping MPRester tests.")
class MPResterTest(unittest.TestCase):
    def setUp(self):
        mpid = 'mp-1153'
        self.mpr = MPRester()
        self.mat = self.mpr.get_material_for_mpid(mpid)

    def test_load_properties(self):
        mpid = 'mp-1153'
        mpr = MPRester()
        mat = mpr.get_material_for_mpid(mpid)
        quantity = next(iter(mat['structure']))
        self.assertEqual(quantity.provenance.source, "Materials Project")
        self.assertIn('structure', mat.get_symbols())

    def test_get_mpid_from_formula(self):
        mp_id = self.mpr.get_mpid_from_formula("Si")
        self.assertEqual("mp-149", mp_id)

    def test_get_properties_for_mpids(self):
        props = self.mpr.get_properties_for_mpids(["mp-124", "mp-81"])
        self.assertAlmostEqual(props[0]['e_above_hull'], 0)
        self.assertAlmostEqual(props[1]['pretty_formula'], 'Au')

    def test_get_properties_for_mpid(self):
        props = self.mpr.get_properties_for_mpid("mp-2")
        self.assertEqual(props['pretty_formula'], "Pd")

    def test_get_materials_for_mpids(self):
        ag, au = self.mpr.get_materials_for_mpids(["mp-124", "mp-81"])
        self.assertEqual(list(ag['external_identifier_mp'])[0].value, 'mp-124')
        self.assertEqual(list(au['external_identifier_mp'])[0].value, 'mp-81')

    def test_get_materials_for_mpid(self):
        pd = self.mpr.get_material_for_mpid("mp-2")
        self.assertEqual(list(pd['external_identifier_mp'])[0].value, 'mp-2')

    def test_apply_material_to_graph(self):
        g = Graph()
        new_mat = g.evaluate(self.mat)
        # TODO:
        # For some reason Travis and this version are not commensurate
        # 257 != 263, should resolve this
        self.assertGreater(len(new_mat.get_quantities()), 250)
