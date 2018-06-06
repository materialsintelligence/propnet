import unittest

from propnet.ext.matproj import *


class MatProjTest(unittest.TestCase):

    def test_load_properties(self):

        mpid = 'mp-1153'
        mpr = MPRester()
        mat = mpr.get_material_for_mpid(mpid)

        self.assertIn('structure', mat.get_symbols())