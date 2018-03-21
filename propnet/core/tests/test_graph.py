import unittest
from propnet.core.graph import *
from propnet.core.materials import *
from propnet.core.symbols import *
from propnet.core.models import *

class GraphTest(unittest.TestCase):

    def setUp(self):
        self.p = Propnet()

    def test_graph_construction(self):
        self.assertGreaterEqual(Propnet().graph.number_of_nodes(), 1)

    def test_valid_node_types(self):
        print(self.p.graph.nodes)
        # if any node on the graph is not of type Node, raise an error
        for node in self.p.graph.nodes:
            self.assertTrue(isinstance(node, PropnetNode))

    @staticmethod
    def check_graph_symbols(to_test, values: list, node_type: str):
        """
        Checks a graph instance (toTest) to see if it contains corresponding elements.
        Cannot contain duplicates (unless they appear in values) or more than are indicated in the list.
        Args:
            to_test: (networkx.multiDiGraph) graph instance to check.
            values: (list<id>) list of all node_type types that should be present in the graph.
            node_type: (str) type of node being checked (ie. Symbol vs. SymbolType)
        Returns: (bool) indicating whether the graph passed or not.
        """
        checked = list()
        to_check = list(filter(lambda n: n.node_type == PropnetNodeType[node_type], to_test.nodes))
        for checking in to_check:
            v_check = checking.node_value
            v_count = 0
            c_count = 0
            for value in values:
                if value == v_check:
                    v_count += 1
            for value in checked:
                if value == c_count:
                    c_count += 1
            if c_count >= v_count:
                # Graph contains too many of a value.
                return False
            checked.append(v_check)
        for checking in checked:
            v_check = checking
            v_count = 0
            c_count = 0
            for value in values:
                if value == v_check:
                    v_count += 1
            for value in checked:
                if value == v_check:
                    c_count += 1
            if v_count != c_count:
                # Graph is missing value(s).
                return False
        return True

    def testSingleMaterialDegeneratePropertySinglePropagationProperties(self):
        """
        Graph has one material on it: mat1
            mat1 has trivial degenerate properties relative permittivity and relative permeability
                2 experimental relative permittivity measurements
                2 experimental relative permeability measurements
        Determines if TestGraph1 is correctly evaluated using the evaluate method.
        We expect 4 refractive_index properties to be calculated as the following:
            sqrt(3), sqrt(5), sqrt(6), sqrt(10)

        """
        # Setup
        propnet = Propnet()
        mat1 = Material()
        mat1.add_property(Symbol(SymbolType['relative_permeability'], 1, None))
        mat1.add_property(Symbol(SymbolType['relative_permeability'], 2, None))
        mat1.add_property(Symbol(SymbolType['relative_permittivity'], 3, None))
        mat1.add_property(Symbol(SymbolType['relative_permittivity'], 5, None))
        propnet.add_material(mat1)

        propnet.evaluate(material=mat1)

        # Expected outputs
        s_outputs = []
        s_outputs.append(Symbol(SymbolType['relative_permeability'], 1, None))
        s_outputs.append(Symbol(SymbolType['relative_permeability'], 2, None))
        s_outputs.append(Symbol(SymbolType['relative_permittivity'], 3, None))
        s_outputs.append(Symbol(SymbolType['relative_permittivity'], 5, None))
        s_outputs.append(Symbol(SymbolType['refractive_index'], 3 ** 0.5, None))
        s_outputs.append(Symbol(SymbolType['refractive_index'], 5 ** 0.5, None))
        s_outputs.append(Symbol(SymbolType['refractive_index'], 6 ** 0.5, None))
        s_outputs.append(Symbol(SymbolType['refractive_index'], 10 ** 0.5, None))

        m_outputs = [mat1]

        st_outputs = []
        st_outputs.append(SymbolType['relative_permeability'])
        st_outputs.append(SymbolType['relative_permittivity'])
        st_outputs.append(SymbolType['refractive_index'])

        # Test
        self.assertTrue(GraphTest.check_graph_symbols(mat1.graph, s_outputs, 'Symbol'))
        self.assertTrue(GraphTest.check_graph_symbols(mat1.graph, m_outputs, 'Material'))
        self.assertTrue(GraphTest.check_graph_symbols(mat1.graph, st_outputs, 'SymbolType'))

        self.assertTrue(GraphTest.check_graph_symbols(propnet.graph, s_outputs, 'Symbol'))
        self.assertTrue(GraphTest.check_graph_symbols(propnet.graph, m_outputs, 'Material'))

    def testDoubleMaterialNonDegeneratePropertySinglePropagationProperties(self):
        """
        Graph has two materials on it: mat1 & mat2
            mat1 has nondegenerate properties relative permittivity and relative permeability
            mat2 has nondegenerate properties relative permittivity and relative permeability
        Determines if TestGraph1 is correctly evaluated using the evaluate method.
        We expect 4 refractive_index properties to be calculated as the following:
            sqrt(3), sqrt(5), sqrt(6), sqrt(10)
        We expect each material graph to have 2 include refractive_index nodes.
        We expect each property jointly calculated to have a joint_material tag element.
        """
        # Setup
        propnet = Propnet()
        mat1 = Material()
        mat2 = Material()
        mat1.add_property(Symbol(SymbolType['relative_permeability'], 1, None))
        mat2.add_property(Symbol(SymbolType['relative_permeability'], 2, None))
        mat1.add_property(Symbol(SymbolType['relative_permittivity'], 3, None))
        mat2.add_property(Symbol(SymbolType['relative_permittivity'], 5, None))
        propnet.add_material(mat1)
        propnet.add_material(mat2)

        propnet.evaluate()

        # Expected propnet outputs
        s_outputs = []
        s_outputs.append(Symbol(SymbolType['relative_permeability'], 1, None))
        s_outputs.append(Symbol(SymbolType['relative_permeability'], 2, None))
        s_outputs.append(Symbol(SymbolType['relative_permittivity'], 3, None))
        s_outputs.append(Symbol(SymbolType['relative_permittivity'], 5, None))
        s_outputs.append(Symbol(SymbolType['refractive_index'], 3 ** 0.5, None))
        s_outputs.append(Symbol(SymbolType['refractive_index'], 5 ** 0.5, None))
        s_outputs.append(Symbol(SymbolType['refractive_index'], 6 ** 0.5, None))
        s_outputs.append(Symbol(SymbolType['refractive_index'], 10 ** 0.5, None))

        m_outputs = [mat1, mat2]

        # Expected mat_1 outputs
        m1_s_outputs = []
        m1_s_outputs.append(Symbol(SymbolType['relative_permeability'], 1, None))
        m1_s_outputs.append(Symbol(SymbolType['relative_permittivity'], 3, None))
        m1_s_outputs.append(Symbol(SymbolType['refractive_index'], 3 ** 0.5, None))

        #Expected mat_2 outputs
        m2_s_outputs = []
        m2_s_outputs.append(Symbol(SymbolType['relative_permeability'], 2, None))
        m2_s_outputs.append(Symbol(SymbolType['relative_permittivity'], 5, None))
        m2_s_outputs.append(Symbol(SymbolType['refractive_index'], 10 ** 0.5, None))

        st_outputs = []
        st_outputs.append(SymbolType['relative_permeability'])
        st_outputs.append(SymbolType['relative_permittivity'])
        st_outputs.append(SymbolType['refractive_index'])

        # Test
        self.assertTrue(GraphTest.check_graph_symbols(mat1.graph, m1_s_outputs, 'Symbol'))
        self.assertTrue(GraphTest.check_graph_symbols(mat1.graph, [mat1], 'Material'))
        self.assertTrue(GraphTest.check_graph_symbols(mat1.graph, st_outputs, 'SymbolType'))

        self.assertTrue(GraphTest.check_graph_symbols(mat2.graph, m2_s_outputs, 'Symbol'))
        self.assertTrue(GraphTest.check_graph_symbols(mat2.graph, [mat2], 'Material'))
        self.assertTrue(GraphTest.check_graph_symbols(mat2.graph, st_outputs, 'SymbolType'))

        self.assertTrue(GraphTest.check_graph_symbols(propnet.graph, s_outputs, 'Symbol'))
        self.assertTrue(GraphTest.check_graph_symbols(propnet.graph, m_outputs, 'Material'))

    def testDoubleMaterialNonDegeneratePropertyDoublePropagationProperies(self):
        """
        Graph has two materials on it mat1 and mat2.
            mat1 has nondegenerate properties A=2 & B=3
            mat2 has nondegenerate properties B=5 & C=7

        Graph has two models on it:
            model1 takes in properties A & B to produce property C=A*B
            model2 takes in properties C & B to produce property E=C/B

        We expect propagate to create a graph structure as follows:
            mat1 has properties: C=6, E=2
            mat2 has properties: E=7/5
            Joint properties: E=6/5, E=7/3
        """
        # Setup
        A = SymbolMetadata('A', [], ['A'], ['A'], [1], np.asarray([1]), '', strict=False)
        B = SymbolMetadata('B', [], ['B'], ['B'], [1], np.asarray([1]), '', strict=False)
        C = SymbolMetadata('C', [], ['C'], ['C'], [1], np.asarray([1]), '', strict=False)
        E = SymbolMetadata('E', [], ['E'], ['E'], [1], np.asarray([1]), '', strict=False)

        mat1 = Material()
        mat2 = Material()
        mat1.add_property(Symbol(A, 2, []))
        mat1.add_property(Symbol(B, 3, []))
        mat2.add_property(Symbol(B, 5, []))
        mat2.add_property(Symbol(C, 7, []))

        class CustomModel(AbstractModel):
            def __init__(self, metadata):
                super(CustomModel, self).__init__(metadata=metadata)

        model1 = CustomModel(metadata={
            'title': 'model1',
            'tags': [],
            'references': [],
            'symbol_mapping': {'a': 'A',
                               'b': 'B',
                               'c': 'C'},
            'connections': [{'inputs': ['c', 'b'],
                             'outputs':['a']}],
            'equations': ['c = a*b'],
            'description': ''
        })

        model2 = CustomModel(metadata={
            'title': 'model2',
            'tags': [],
            'references': [],
            'symbol_mapping': {'e': 'E',
                               'b': 'B',
                               'c': 'C'},
            'connections': [{'inputs': ['c', 'b'],
                             'outputs': ['e']}],
            'equations': ['e = c/b'],
            'description': ''
        })

        p = Propnet(auto_load=False)
        p.add_property_types([A, B, C, E])
        p.add_models([model1])

        print(p)