import unittest
from propnet.core.graph import *


class GraphTest(unittest.TestCase):
    """ """

    def setUp(self):

        self.p = Propnet()

    def testGraphConstruction(self):

        self.assertGreaterEqual(self.p.graph.number_of_nodes(), 1)

    def testValidNodeTypes(self):

        print(self.p.graph.nodes)

        # if any node on the graph is not of type Node, raise an error
        for node in self.p.graph.nodes:
            if not isinstance(node, PropnetNode):
                raise ValueError('Node on graph is not of valid type: {}'.format(node))