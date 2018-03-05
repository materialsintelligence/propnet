import unittest
from propnet.core.symbols import *
from propnet.symbols import *
from propnet.ext.matproj import *

class MPInterfaceTest(unittest.TestCase):
    """ """

    def testLoadProperties(self):
        mpID = 'mp-1153'
        mat = import_material(mpID)
        self.assertTrue('structure' in mat.available_properties())
        self.assertTrue('lattice_unit_cell' in mat.available_properties())
