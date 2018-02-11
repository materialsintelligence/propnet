import unittest
from propnet.core.graph import *
from propnet.core.materials import *
from propnet.core.symbols import *

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

    def testEvaluateMethod(self):

        def genTestGraph1():
            """
            genTestGraph1 (void) -> Propnet
            Returns a graph used for testing.
                Graph has two materials on it: mat1 & mat2
                mat1 has trivial degenerate properties relative permittivity and relative permeability
                    2 experimental relative permittivity measurements
                    2 experimental relative permeability measurements
            """
            toReturn = Propnet()
            mat1 = Material()
            mat1.add_property(Symbol(SymbolType['relative_permeability'], 1, None))
            mat1.add_property(Symbol(SymbolType['relative_permeability'], 2, None))
            mat1.add_property(Symbol(SymbolType['relative_permittivity'], 3, None))
            mat1.add_property(Symbol(SymbolType['relative_permittivity'], 5, None))
            toReturn.add_material(mat1)
            return toReturn

        def testTestGraph1(propnet: Propnet):
            """
            genTestGraph1 (Propnet) -> void
            Determines if TestGraph1 is correctly evaluated using the evaluate method.
            We expect 4 refractive_index properties to be calculated as the following:
                sqrt(3), sqrt(5), sqrt(6), sqrt(10)
            """
            propnet.evaluate()
            print(propnet)
            materialNodes = propnet.nodes_by_type('Material')
            symbolNodes = propnet.nodes_by_type('Symbol')
            if len(materialNodes) != 1:
                raise ValueError('No material node appears on TestGraph1')
            properties = [n.node_value for n in symbolNodes]
            if len(list(filter(lambda n: n.type == SymbolType['relative_permeability'], properties))) != 2:
                raise ValueError('Missing relative_permeability symbols')
            for item in filter(lambda n: n.type == SymbolType['relative_permeability'], properties):
                if item.value not in [1, 2]:
                    raise ValueError('relative_permeability improperly mutated')
            if len(list(filter(lambda n: n.type == SymbolType['relative_permittivity'], properties))) != 2:
                raise ValueError('Missing relative_permittivity symbols')
            for item in filter(lambda n: n.type == SymbolType['relative_permittivity'], properties):
                if item.value not in [3, 5]:
                    raise ValueError('relative_permeability improperly mutated')
            if len(list(filter(lambda n: n.type == SymbolType['refractive_index'], properties))) != 4:
                raise ValueError('Missing refractive_index nodes.')
            for item in filter(lambda n: n.type == SymbolType['refractive_index'], properties):
                if int(item.value**2 + 0.5) not in [3, 5, 6, 10]:
                    raise ValueError('refractive_index improperly calculated')

        testing1 = genTestGraph1()
        print(testing1)
        testTestGraph1(testing1)
