import unittest
from propnet.core.graph import *


class GraphTest(unittest.TestCase):
    """ """

    def testGraphConstruction(self):
        """ """

        p = Propnet()
        self.assertGreaterEqual(p.graph.number_of_nodes(), 1)