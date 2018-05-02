import unittest
from propnet.core.graph import *
from propnet.core.materials import *
from propnet.core.symbols import *
from propnet.core.models import *

from propnet.symbols import DEFAULT_SYMBOL_TYPES

class GraphTest(unittest.TestCase):

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
                if value == v_check:
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

    def setUp(self):
        self.p = Propnet()

    def test_graph_construction(self):
        self.assertGreaterEqual(self.p.graph.number_of_nodes(), 1)

    def test_valid_node_types(self):
        print(self.p.graph.nodes)
        # if any node on the graph is not of type Node, raise an error
        for node in self.p.graph.nodes:
            self.assertTrue(isinstance(node, PropnetNode))

    def test_add_material(self):
        """
        Adding a material to a Propnet instance should lead to all the
        Symbol, SymbolType, and Material nodes of the material getting
        added to the Propnet instance -- a disjoint union.

        Given a general Propnet instance, we add a material with custom
        SymbolType, A, and two Symbol nodes of type A with different values.

        Returns: None
        """
        # Setup
        p = Propnet()
        A = SymbolType('a', ['A'], ['A'], units=[1.0, []], dimension=[1])
        mat1 = Material()
        mat1.add_property(Symbol(A, 2, []))
        mat1.add_property(Symbol(A, 3, []))
        mat2 = Material()
        mat2.add_property(Symbol(A, 4, []))
        p.add_material(mat2)

        # Add Material
        p.add_material(mat1)

        # Test Graph
        # 1) Material's graph should be unchanged.
        # 2) Propnet instance should be appropriately updated.

        self.assertTrue(GraphTest.check_graph_symbols(mat1.graph, [Symbol(A, 2, []), Symbol(A, 3, [])], 'Symbol'))
        self.assertTrue(GraphTest.check_graph_symbols(mat1.graph, [A], 'SymbolType'))
        self.assertTrue(GraphTest.check_graph_symbols(mat1.graph, [mat1], 'Material'))

        self.assertTrue(GraphTest.check_graph_symbols(
            p.graph, [Symbol(A, 2, []), Symbol(A, 3, []), Symbol(A, 4, [])], 'Symbol'))
        self.assertTrue(GraphTest.check_graph_symbols(
            p.graph, list(DEFAULT_SYMBOL_TYPES.values()) + [A], 'SymbolType'))
        self.assertTrue(GraphTest.check_graph_symbols(
            p.graph, [mat1, mat2], 'Material'))

    def test_remove_material(self):
        """
        Adding a removing a material from a Propnet instance should lead
        to all the Symbol nodes from the specific material being removed
        from the Propnet instance, along with the material's node. The
        material's graph should not be altered.

        Given a general Propnet instance, we add a material with Symbols.
        We ensure that adding and removing does not alter its graph.
        We ensure that adding and then removing the material does not
        alter the Propnet instance.

        Returns: None
        """
        # Setup
        p = Propnet()
        mat1 = Material()
        mat1.add_property(Symbol('refractive_index', 1, []))
        mat1.add_property(Symbol('relative_permittivity', 2, []))
        mat2 = Material()
        mat2.add_property(Symbol('refractive_index', 1, []))
        p.add_material(mat1)
        p.add_material(mat2)
        p.remove_material(mat1)

        # Test Graph
        self.assertTrue(GraphTest.check_graph_symbols(mat1.graph, [
            Symbol('refractive_index', 1, []), Symbol('relative_permittivity', 2, [])
        ], 'Symbol'))
        self.assertTrue(GraphTest.check_graph_symbols(mat1.graph, [
            Symbol('refractive_index', 1.3, []).type, Symbol('relative_permittivity', 2, []).type
        ], 'SymbolType'))
        self.assertTrue(GraphTest.check_graph_symbols(p.graph, [Symbol('refractive_index', 1, [])], 'Symbol'))
        self.assertTrue(GraphTest.check_graph_symbols(p.graph, [mat2], 'Material'))

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
        mat1.add_property(Symbol(DEFAULT_SYMBOL_TYPES['relative_permeability'], 1, None))
        mat1.add_property(Symbol(DEFAULT_SYMBOL_TYPES['relative_permeability'], 2, None))
        mat1.add_property(Symbol(DEFAULT_SYMBOL_TYPES['relative_permittivity'], 3, None))
        mat1.add_property(Symbol(DEFAULT_SYMBOL_TYPES['relative_permittivity'], 5, None))
        propnet.add_material(mat1)

        propnet.evaluate(material=mat1)

        # Expected outputs
        s_outputs = []
        s_outputs.append(Symbol('relative_permeability', 1, None))
        s_outputs.append(Symbol('relative_permeability', 2, None))
        s_outputs.append(Symbol('relative_permittivity', 3, None))
        s_outputs.append(Symbol('relative_permittivity', 5, None))
        s_outputs.append(Symbol('refractive_index', 3 ** 0.5, None))
        s_outputs.append(Symbol('refractive_index', 5 ** 0.5, None))
        s_outputs.append(Symbol('refractive_index', 6 ** 0.5, None))
        s_outputs.append(Symbol('refractive_index', 10 ** 0.5, None))

        m_outputs = [mat1]

        st_outputs = []
        st_outputs.append(DEFAULT_SYMBOL_TYPES['relative_permeability'])
        st_outputs.append(DEFAULT_SYMBOL_TYPES['relative_permittivity'])
        st_outputs.append(DEFAULT_SYMBOL_TYPES['refractive_index'])

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
        mat1.add_property(Symbol('relative_permeability', 1, None))
        mat2.add_property(Symbol('relative_permeability', 2, None))
        mat1.add_property(Symbol('relative_permittivity', 3, None))
        mat2.add_property(Symbol('relative_permittivity', 5, None))
        propnet.add_material(mat1)
        propnet.add_material(mat2)

        propnet.evaluate()

        # Expected propnet outputs
        s_outputs = []
        s_outputs.append(Symbol('relative_permeability', 1, None))
        s_outputs.append(Symbol('relative_permeability', 2, None))
        s_outputs.append(Symbol('relative_permittivity', 3, None))
        s_outputs.append(Symbol('relative_permittivity', 5, None))
        s_outputs.append(Symbol('refractive_index', 3 ** 0.5, None))
        s_outputs.append(Symbol('refractive_index', 5 ** 0.5, None))
        s_outputs.append(Symbol('refractive_index', 6 ** 0.5, None))
        s_outputs.append(Symbol('refractive_index', 10 ** 0.5, None))

        m_outputs = [mat1, mat2]

        # Expected mat_1 outputs
        m1_s_outputs = []
        m1_s_outputs.append(Symbol('relative_permeability', 1, None))
        m1_s_outputs.append(Symbol('relative_permittivity', 3, None))
        m1_s_outputs.append(Symbol('refractive_index', 3 ** 0.5, None))

        #Expected mat_2 outputs
        m2_s_outputs = []
        m2_s_outputs.append(Symbol('relative_permeability', 2, None))
        m2_s_outputs.append(Symbol('relative_permittivity', 5, None))
        m2_s_outputs.append(Symbol('refractive_index', 10 ** 0.5, None))

        st_outputs = []
        st_outputs.append(DEFAULT_SYMBOL_TYPES['relative_permeability'])
        st_outputs.append(DEFAULT_SYMBOL_TYPES['relative_permittivity'])
        st_outputs.append(DEFAULT_SYMBOL_TYPES['refractive_index'])

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

        a = SymbolType('a', ['A'], ['A'], units=[1.0, []], dimension=[1])
        b = SymbolType('b', ['A'], ['A'], units=[1.0, []], dimension=[1])
        c = SymbolType('c', ['A'], ['A'], units=[1.0, []], dimension=[1])
        e = SymbolType('e', ['A'], ['A'], units=[1.0, []], dimension=[1])
        symbol_type_dict = {'a': a, 'b': b, 'c': c, 'e': e}

        mat1 = Material()
        mat2 = Material()
        mat1.add_property(Symbol(a, 2, []))
        mat1.add_property(Symbol(b, 3, []))
        mat2.add_property(Symbol(b, 5, []))
        mat2.add_property(Symbol(c, 7, []))

        class Model1 (AbstractModel):
            def __init__(self, symbol_types=None):
                AbstractModel.__init__(self, metadata={
                        'title': 'model1',
                        'tags': [],
                        'references': [],
                        'symbol_mapping': {'a': 'a',
                                           'b': 'b',
                                           'c': 'c'
                                           },
                        'connections': [{
                                         'inputs': ['a', 'b'],
                                         'outputs': ['c']
                                         }],
                        'equations': ['c=a*b'],
                        'description': ''
                    },
                    symbol_types=symbol_types)

        class Model2 (AbstractModel):
            def __init__(self, symbol_types=None):
                AbstractModel.__init__(self, metadata={
                        'title': 'model2',
                        'tags': [],
                        'references': [],
                        'symbol_mapping': {'e': 'e',
                                           'c': 'c',
                                           'b': 'b'},
                        'connections': [{'inputs': ['c', 'b'],
                                         'outputs': ['e']
                                         }],
                        'equations': ['e=c/b'],
                        'description': ''
                    },
                    symbol_types=symbol_types)

        p = Propnet(materials=[mat1, mat2],
                    models={'model1': Model1, 'model2': Model2},
                    symbol_types=symbol_type_dict)

        # Evaluate
        p.evaluate()

        # Test
        m1_s_outputs = []
        m1_s_outputs.append(Symbol(a, 2, []))
        m1_s_outputs.append(Symbol(b, 3, []))
        m1_s_outputs.append(Symbol(c, 6, []))
        m1_s_outputs.append(Symbol(e, 2, []))

        m2_s_outputs = []
        m2_s_outputs.append(Symbol(b, 5, []))
        m2_s_outputs.append(Symbol(c, 7, []))
        m2_s_outputs.append(Symbol(e, 7/5, []))

        joint_outputs = []
        joint_outputs.append(Symbol(e, 6/5, []))
        joint_outputs.append(Symbol(e, 7/3, []))

        self.assertTrue(GraphTest.check_graph_symbols(mat1.graph, m1_s_outputs, 'Symbol'))
        self.assertTrue(GraphTest.check_graph_symbols(mat1.graph, [mat1], 'Material'))
        self.assertTrue(GraphTest.check_graph_symbols(mat1.graph, [a, b, c, e], 'SymbolType'))

        self.assertTrue(GraphTest.check_graph_symbols(mat2.graph, m2_s_outputs, 'Symbol'))
        self.assertTrue(GraphTest.check_graph_symbols(mat2.graph, [mat2], 'Material'))
        self.assertTrue(GraphTest.check_graph_symbols(mat2.graph, [b, c, e], 'SymbolType'))

        self.assertTrue(GraphTest.check_graph_symbols(
            p.graph, m1_s_outputs + m2_s_outputs + joint_outputs, 'Symbol'))
        self.assertTrue(GraphTest.check_graph_symbols(
            p.graph, [a, b, c, e], 'SymbolType'))
        self.assertTrue(GraphTest.check_graph_symbols(
            p.graph, [mat1, mat2], 'Material'))

    def testEvaluationWithConstraint(self):
        """
        Simple graph in which property C can be derived from properties A and B iff Constraint has value True.
        Graph has 2 materials on it: mat1 and mat2.
            mat1 has a Constraint with value True.
            mat2 has a Constraint with value False.
        Upon evaluation of the graph, we expect c to be derived ONLY for mat1.
        """

        # Setup

        a = SymbolType('a', ['A'], ['A'], units=[1.0, []], dimension=[1])
        b = SymbolType('b', ['A'], ['A'], units=[1.0, []], dimension=[1])
        c = SymbolType('c', ['A'], ['A'], units=[1.0, []], dimension=[1])
        constraint = SymbolType('constraint', ['C'], ['C'], category='object')
        symbol_type_dict = {'a': a, 'b': b, 'c': c, 'constraint': constraint}

        a_example = Symbol(a, 2, [])
        b_example = Symbol(b, 3, [])
        const1 = Symbol(constraint, True, [])
        const2 = Symbol(constraint, False, [])

        class Model1 (AbstractModel):
            def __init__(self, symbol_types=None):
                AbstractModel.__init__(self, metadata={
                        'title': 'model1',
                        'tags': [],
                        'references': [],
                        'symbol_mapping': {'a': 'a',
                                           'b': 'b',
                                           'c': 'c',
                                           'const': 'constraint'
                                           },
                        'connections': [{
                                         'inputs': ['a', 'b'],
                                         'outputs': ['c']
                                         }],
                        'equations': ['c=a*b'],
                        'description': ''
                    },
                    symbol_types=symbol_types)
            @property
            def constraint_symbols(self):
                return ['const']
            def check_constraints(self, ins):
                return ins['const']

        mat1 = Material()
        mat1.add_property(a_example)
        mat1.add_property(b_example)
        mat1.add_property(const1)

        mat2 = Material()
        mat2.add_property(a_example)
        mat2.add_property(b_example)
        mat2.add_property(const2)

        p = Propnet(materials=[mat1, mat2],
                    models={'model1': Model1},
                    symbol_types=symbol_type_dict)

        # Evaluate
        p.evaluate(material=mat1)
        p.evaluate(material=mat2)

        # Test
        m1_s_outputs = []
        m1_s_outputs.append(Symbol(a, 2, []))
        m1_s_outputs.append(Symbol(b, 3, []))
        m1_s_outputs.append(Symbol(c, 6, []))
        m1_s_outputs.append(Symbol(constraint, True, []))

        m2_s_outputs = []
        m2_s_outputs.append(Symbol(a, 2, []))
        m2_s_outputs.append(Symbol(b, 3, []))
        m2_s_outputs.append(Symbol(constraint, False, []))

        self.assertTrue(GraphTest.check_graph_symbols(mat1.graph, m1_s_outputs, 'Symbol'))
        self.assertTrue(GraphTest.check_graph_symbols(mat1.graph, [mat1], 'Material'))
        self.assertTrue(GraphTest.check_graph_symbols(mat1.graph, [a, b, c, constraint], 'SymbolType'))

        self.assertTrue(GraphTest.check_graph_symbols(mat2.graph, m2_s_outputs, 'Symbol'))
        self.assertTrue(GraphTest.check_graph_symbols(mat2.graph, [mat2], 'Material'))
        self.assertTrue(GraphTest.check_graph_symbols(mat2.graph, [a, b, constraint], 'SymbolType'))
