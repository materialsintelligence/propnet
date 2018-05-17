import unittest
from propnet.core.symbols import *
from propnet.symbols import *
from propnet.ext.matproj import *


class MatProjTest(unittest.TestCase):

    def test_load_properties(self):

        mp_id = 'mp-1153'
        mat = import_material(mp_id)
        self.assertTrue('structure' in mat.available_properties())
        self.assertTrue('lattice_unit_cell' in mat.available_properties())