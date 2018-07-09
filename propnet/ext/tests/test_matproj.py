import unittest

from propnet.ext.matproj import MPRester


class MPResterTest(unittest.TestCase):

    def test_load_properties(self):
        mpid = 'mp-1153'
        mpr = MPRester()
        mat = mpr.get_material_for_mpid(mpid)
        self.assertIn('structure', mat.get_symbols())

    def test_get_mpid_from_formula(self):
        pass

    def test_get_properties_for_mpids(self):
        pass

    def test_get_properties_for_mpid(self):
        pass

    def test_get_materials_for_mpids(self):
        pass

    def test_get_materials_for_mpid(self):
        pass