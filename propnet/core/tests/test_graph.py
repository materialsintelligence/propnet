import unittest
from propnet.core.graph import Graph, SymbolPath
from propnet.core.materials import Material
from propnet.core.symbols import Symbol
from propnet.core.models import AbstractModel
from propnet.core.quantity import Quantity

from propnet.symbols import DEFAULT_SYMBOLS

class GraphTest(unittest.TestCase):

    @staticmethod
    def generate_canonical_symbols():
        """
        Returns a set of Symbol objects used in testing.
        Returns: (dict<str, Symbol>)
        """
        A = Symbol('A', ['A'], ['A'], units=[1.0, []], shape=[1])
        B = Symbol('B', ['B'], ['B'], units=[1.0, []], shape=[1])
        C = Symbol('C', ['C'], ['C'], units=[1.0, []], shape=[1])
        D = Symbol('D', ['D'], ['D'], units=[1.0, []], shape=[1])
        G = Symbol('G', ['G'], ['G'], units=[1.0, []], shape=[1])
        F = Symbol('F', ['F'], ['F'], units=[1.0, []], shape=[1])
        return {
            'A': A,
            'B': B,
            'C': C,
            'D': D,
            'G': G,
            'F': F
        }

    @staticmethod
    def generate_canonical_models(c_symbols):
        """
        Returns a set of Model objects used in testing.
        Returns: (dict<str, Model>)
        """

        class Model1 (AbstractModel):
            def __init__(self, symbol_types=None):
                AbstractModel.__init__(self, metadata=
                    { 'title': 'model1', 'tags': [], 'references': [], 'description': '',
                      'symbol_mapping': {'A': 'A',
                                         'B': 'B',
                                         'C': 'C'
                                        },
                      'connections': [{'inputs': ['A'],
                                       'outputs': ['B', 'C']
                                      }],
                      'equations': ['B-2*A', 'C-3*A']
                    },
                    symbol_types=symbol_types)

        class Model2 (AbstractModel):
            def __init__(self, symbol_types=None):
                AbstractModel.__init__(self, metadata=
                    { 'title': 'model2', 'tags': [], 'references': [], 'description': '',
                      'symbol_mapping': {'A': 'A',
                                         'G': 'G'
                                        },
                      'connections': [{'inputs': ['A'],
                                       'outputs': ['G']
                                      }],
                      'equations': ['G-5*A']
                    },
                    symbol_types=symbol_types)

        class Model3 (AbstractModel):
            def __init__(self, symbol_types=None):
                AbstractModel.__init__(self, metadata=
                    { 'title': 'model3', 'tags': [], 'references': [], 'description': '',
                      'symbol_mapping': {'B': 'B',
                                         'F': 'F'
                                        },
                      'connections': [{'inputs': ['B'],
                                       'outputs': ['F']
                                      }],
                      'equations': ['F-7*B']
                    },
                    symbol_types=symbol_types)

        class Model4 (AbstractModel):
            def __init__(self, symbol_types=None):
                AbstractModel.__init__(self, metadata=
                    { 'title': 'model4', 'tags': [], 'references': [], 'description': '',
                      'symbol_mapping': {'D': 'D',
                                         'B': 'B',
                                         'C': 'C'
                                        },
                      'connections': [{'inputs': ['B', 'C'],
                                       'outputs': ['D']
                                      }],
                      'equations': ['D-B*C*11']
                    },
                    symbol_types=symbol_types)

        class Model5 (AbstractModel):
            def __init__(self, symbol_types=None):
                AbstractModel.__init__(self, metadata=
                    { 'title': 'model5', 'tags': [], 'references': [], 'description': '',
                      'symbol_mapping': {'C': 'C',
                                         'G': 'G',
                                         'D': 'D'
                                        },
                      'connections': [{'inputs': ['C', 'G'],
                                       'outputs': ['D']
                                      }],
                      'equations': ['D-C*G*13']
                    },
                    symbol_types=symbol_types)

        class Model6 (AbstractModel):
            def __init__(self, symbol_types=None):
                AbstractModel.__init__(self, metadata=
                    { 'title': 'model6', 'tags': [], 'references': [], 'description': '',
                      'symbol_mapping': {'A': 'A',
                                         'F': 'F',
                                         'D': 'D'
                                        },
                      'connections': [{'inputs': ['F', 'D'],
                                       'outputs': ['A']
                                      }],
                      'equations': ['A-F*D*17']
                    },
                    symbol_types=symbol_types)

        models = [Model1(c_symbols), Model2(c_symbols), Model3(c_symbols),
                  Model4(c_symbols), Model5(c_symbols), Model6(c_symbols)]
        return {x.title: x for x in models}

    @staticmethod
    def generate_canonical_material(c_symbols):
        """
        Generates a Material with appropriately attached Quantities.
        Args:
            c_symbols: (dict<str, Symbol>) dictionary of defined materials.
        Returns:
            (Material) material with properties loaded.
        """
        q1 = Quantity(c_symbols['A'], 19)
        q2 = Quantity(c_symbols['A'], 23)
        m = Material()
        m.add_quantity(q1)
        m.add_quantity(q2)
        return m

    def test_graph_setup(self):
        """
        Tests the outcome of constructing the canonical graph.
        """
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(symbols)
        g = Graph(models=models, symbol_types=symbols)
        st_c = {x for x in symbols.values()}
        st_g = g.get_symbol_types()
        m_c = {x for x in models.values()}
        m_g = g.get_models()
        self.assertTrue(st_c == st_g,
                        'Canonical constructed graph does not have the right Symbol objects.')
        self.assertTrue(m_c == m_g,
                        'Canonical constructed graph does not have the right Model objects.')
        for m in models.values():
            for d in m.type_connections:
                for s in d['inputs']:
                    self.assertTrue(symbols[s] in g._input_to_model.keys(),
                                    "Canonical constructed graph does not have an edge from input: " + s +
                                    " to model: " + m.name)
                    self.assertTrue(m in g._input_to_model[s],
                                    "Canonical constructed graph does not have an edge from input: " + s +
                                    " to model: " + m.name)
                for s in d['outputs']:
                    self.assertTrue(symbols[s] in g._output_to_model.keys(),
                                    "Canonical constructed graph does not have an edge from input: " + s +
                                    " to model: " + m.name)
                    self.assertTrue(m in g._output_to_model[s],
                                    "Canonical constructed graph does not have an edge from input: " + s +
                                    " to model: " + m.name)

    def test_model_add_remove(self):
        """
        Tests the outcome of adding and removing a model from the canonical graph.
        """
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(symbols)
        g = Graph(models=models, symbol_types=symbols)
        g.remove_models({models['model6'].name: models['model6']})
        self.assertTrue(models['model6'] not in g.get_models(),
                        "Model was unsuccessfully removed from the graph.")
        for s in g._input_to_model.values():
            self.assertTrue(models['model6'] not in s,
                            "Model was unsuccessfully removed from the graph.")
        for s in g._output_to_model.values():
            self.assertTrue(models['model6'] not in s,
                            "Model was unsuccessfully removed from the graph.")
        m6 = models['model6']
        del models['model6']
        for m in models.values():
            self.assertTrue(m in g.get_models(),
                            "Too many models were removed.")
        g.update_models({'Model6': m6})
        self.assertTrue(m6 in g.get_models(),
                        "Model was unsuccessfully added to the graph.")
        self.assertTrue(m6 in g._input_to_model[symbols['D']],
                        "Model was unsuccessfully added to the graph.")
        self.assertTrue(m6 in g._input_to_model[symbols['F']],
                        "Model was unsuccessfully added to the graph.")
        self.assertTrue(m6 in g._output_to_model[symbols['A']],
                        "Model was unsuccessfully added to the graph.")

    def test_symbol_add_remove(self):
        """
        Tests the outcome of adding and removing a Symbol from the canonical graph.
        """
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(symbols)
        g = Graph(models=models, symbol_types=symbols)
        g.remove_symbol_types({'F': symbols['F']})
        self.assertTrue(symbols['F'] not in g.get_symbol_types(),
                        "Symbol was not properly removed.")
        self.assertTrue(symbols['F'] not in g._input_to_model.keys(),
                        "Symbol was not properly removed.")
        self.assertTrue(symbols['F'] not in g._output_to_model.keys(),
                        "Symbol was not properly removed.")
        self.assertTrue(models['model3'] not in g.get_models(),
                        "Removing symbol did not remove a model using that symbol.")
        self.assertTrue(models['model6'] not in g.get_models(),
                        "Removing symbol did not remove a model using that symbol.")
        g.update_symbol_types({'F': symbols['F']})
        self.assertTrue(symbols['F'] in g.get_symbol_types(),
                       "Symbol was not properly added.")

    def test_add_remove_material(self):
        """
        Tests the outcome of adding and removing a Material from the canonical graph.
        """
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(symbols)
        g = Graph(models=models, symbol_types=symbols)
        m = GraphTest.generate_canonical_material(symbols)
        g.add_material(m)
        qs = m.get_quantities()
        self.assertTrue(m in g.get_materials(),
                        "Material improperly added to the graph.")
        self.assertTrue(symbols['A'] in g._symbol_to_quantity,
                        "Material improperly added to the graph.")
        for q in qs:
            self.assertTrue(q in g._symbol_to_quantity[q.symbol],
                            "Material quantities improperly added to the graph.")
        g.remove_material(m)
        self.assertTrue(len(g.get_materials()) == 0,
                        "Material improperly removed from the graph.")
        self.assertTrue(len(g._symbol_to_quantity) == 0,
                        "Material improperly removed from the graph.")

    def test_network_X(self):
        """
        Tests the outcome of calculating the networkx graph datastructure.
        """
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(symbols)
        g = Graph(models=models, symbol_types=symbols)
        m = GraphTest.generate_canonical_material(symbols)
        g.add_material(m)
        qs = m.get_quantities()
        nx = g.graph
        t1 = [x for x in nx.predecessors(symbols['A'])]
        self.assertTrue(models['model6'] in t1,
                        "Graph improperly constructed.")
        for q in qs:
            self.assertTrue(q in t1,
                            "Graph improperly constructed.")
        t2 = [x for x in nx.successors(symbols['A'])]
        self.assertTrue(models['model1'] in t2,
                        "Graph improperly constructed.")
        self.assertTrue(models['model2'] in t2,
                        "Graph improperly constructed.")
        t3 = [x for x in nx.predecessors(qs[0])]
        self.assertTrue(m in t3,
                        "Graph improperly constructed.")
        t4 = [x for x in nx.successors(models['model1'])]
        self.assertTrue(symbols['B'] in t4,
                        "Graph improperly constructed.")
        self.assertTrue(symbols['C'] in t4,
                        "Graph improperly constructed.")
        t5 = [x for x in nx.predecessors(models['model1'])]
        self.assertTrue(symbols['A'] in t5,
                        "Graph improperly constructed.")

    def test_add_remove_material_quantity(self):
        """
        Tests adding or removing a quantity from a material attached to a graph.
        """
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(symbols)
        g = Graph(models=models, symbol_types=symbols)
        m = GraphTest.generate_canonical_material(symbols)
        g.add_material(m)
        qs = m.get_quantities()
        q1 = qs[0]
        q2 = qs[1]
        m.remove_quantity(q1)
        self.assertTrue(q1 not in g._symbol_to_quantity[symbols['A']],
                        "Quantity was unsuccessfully removed.")
        self.assertTrue(q2 in g._symbol_to_quantity[symbols['A']],
                        "Extra Quantity was removed.")
        m.add_quantity(q1)
        self.assertTrue(q1 in g._symbol_to_quantity[symbols['A']],
                        "Quantity was unsuccessfully added.")

    def test_evaluate(self):
        """
        Tests the evaluation algorithm on a non-cyclic graph.
        The canonical graph and the canonical material are used for this test.
        """
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(symbols)
        material = GraphTest.generate_canonical_material(symbols)
        del models['model6']
        g = Graph(materials=[material], symbol_types=symbols, models=models)
        g.evaluate()

        expected_quantities = [
            Quantity(symbols['A'], 19, {material}),
            Quantity(symbols['A'], 23, {material}),
            Quantity(symbols['B'], 38, {material}),
            Quantity(symbols['B'], 46, {material}),
            Quantity(symbols['C'], 57, {material}),
            Quantity(symbols['C'], 69, {material}),
            Quantity(symbols['G'], 95, {material}),
            Quantity(symbols['G'], 115, {material}),
            Quantity(symbols['F'], 266, {material}),
            Quantity(symbols['F'], 322, {material}),
            Quantity(symbols['D'], 23826, {material}),
            Quantity(symbols['D'], 28842, {material}),
            Quantity(symbols['D'], 28842, {material}),
            Quantity(symbols['D'], 34914, {material}),
            Quantity(symbols['D'], 70395, {material}),
            Quantity(symbols['D'], 85215, {material}),
            Quantity(symbols['D'], 85215, {material}),
            Quantity(symbols['D'], 103155, {material}),
        ]

        for q in expected_quantities:
            self.assertTrue(q in g._symbol_to_quantity[q.symbol],
                            "Evaluate failed to derive all outputs.")
            self.assertTrue(material in q._material,
                            "Evaluate failed to assign material.")

    def test_evaluate_cyclic(self):
        """
        Tests the evaluation algorithm on a cyclic graph.
        The canonical graph and the canonical material are used for this test.
        """
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(symbols)
        material = GraphTest.generate_canonical_material(symbols)
        g = Graph(materials=[material], symbol_types=symbols, models=models)
        g.evaluate()

        expected_quantities = [
            Quantity(symbols['A'], 19, {material}),
            Quantity(symbols['A'], 23, {material}),
            Quantity(symbols['B'], 38, {material}),
            Quantity(symbols['B'], 46, {material}),
            Quantity(symbols['C'], 57, {material}),
            Quantity(symbols['C'], 69, {material}),
            Quantity(symbols['G'], 95, {material}),
            Quantity(symbols['G'], 115, {material}),
            Quantity(symbols['F'], 266, {material}),
            Quantity(symbols['F'], 322, {material}),
            Quantity(symbols['D'], 23826, {material}),
            Quantity(symbols['D'], 28842, {material}),
            Quantity(symbols['D'], 28842, {material}),
            Quantity(symbols['D'], 34914, {material}),
            Quantity(symbols['D'], 70395, {material}),
            Quantity(symbols['D'], 85215, {material}),
            Quantity(symbols['D'], 85215, {material}),
            Quantity(symbols['D'], 103155, {material}),
            Quantity(symbols['A'], 107741172, {material}),
            Quantity(symbols['A'], 130423524, {material}),
            Quantity(symbols['A'], 130423524, {material}),
            Quantity(symbols['A'], 157881108, {material}),
            Quantity(symbols['A'], 130423524, {material}),
            Quantity(symbols['A'], 157881108, {material}),
            Quantity(symbols['A'], 157881108, {material}),
            Quantity(symbols['A'], 191119236, {material}),
            Quantity(symbols['A'], 318326190, {material}),
            Quantity(symbols['A'], 385342230, {material}),
            Quantity(symbols['A'], 385342230, {material}),
            Quantity(symbols['A'], 466466910, {material}),
            Quantity(symbols['A'], 385342230, {material}),
            Quantity(symbols['A'], 466466910, {material}),
            Quantity(symbols['A'], 466466910, {material}),
            Quantity(symbols['A'], 564670470, {material})
        ]

        for q in expected_quantities:
            self.assertTrue(q in g._symbol_to_quantity[q.symbol],
                            "Evaluate failed to derive all outputs.")
            self.assertTrue(material in q._material,
                            "Evaluate failed to assign material.")

    def test_evaluate_constraints(self):
        """
        Tests the evaluation algorithm on a non-cyclic graph involving constraints.
        The canonical graph and the canonical material are used for this test.
        """
        class Model4 (AbstractModel):
            def __init__(self, symbol_types=None):
                AbstractModel.__init__(self, metadata=
                    { 'title': 'model4', 'tags': [], 'references': [], 'description': '',
                      'symbol_mapping': {'D': 'D',
                                         'B': 'B',
                                         'C': 'C',
                                         'G': 'G'
                                        },
                      'connections': [{'inputs': ['B', 'C'],
                                       'outputs': ['D']
                                      }],
                      'equations': ['D-B*C*11']
                    },
                    symbol_types=symbol_types)

            @property
            def constraint_symbols(self):
                return ['G']

            def check_constraints(self, constraint_inputs):
                return constraint_inputs['G'] == 0

        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(symbols)
        models['model4'] = Model4(symbol_types=symbols)
        del models['model6']
        material = GraphTest.generate_canonical_material(symbols)
        g = Graph(materials=[material], symbol_types=symbols, models=models)
        g.evaluate()

        expected_quantities = [
            Quantity(symbols['A'], 19, {material}),
            Quantity(symbols['A'], 23, {material}),
            Quantity(symbols['B'], 38, {material}),
            Quantity(symbols['B'], 46, {material}),
            Quantity(symbols['C'], 57, {material}),
            Quantity(symbols['C'], 69, {material}),
            Quantity(symbols['G'], 95, {material}),
            Quantity(symbols['G'], 115, {material}),
            Quantity(symbols['F'], 266, {material}),
            Quantity(symbols['F'], 322, {material}),
            Quantity(symbols['D'], 70395, {material}),
            Quantity(symbols['D'], 85215, {material}),
            Quantity(symbols['D'], 85215, {material}),
            Quantity(symbols['D'], 103155, {material}),
        ]

        for q in expected_quantities:
            self.assertTrue(q in g._symbol_to_quantity[q.symbol],
                            "Evaluate failed to derive all outputs.")
            self.assertTrue(material in q._material,
                            "Evaluate failed to assign material.")

    def test_evaluate_constraints_cyclic(self):
        """
        Tests the evaluation algorithm on a cyclic graph involving constraints.
        The canonical graph and the canonical material are used for this test.
        """
        class Model4 (AbstractModel):
            def __init__(self, symbol_types=None):
                AbstractModel.__init__(self, metadata=
                    { 'title': 'model4', 'tags': [], 'references': [], 'description': '',
                      'symbol_mapping': {'D': 'D',
                                         'B': 'B',
                                         'C': 'C',
                                         'G': 'G'
                                        },
                      'connections': [{'inputs': ['B', 'C'],
                                       'outputs': ['D']
                                      }],
                      'equations': ['D-B*C*11']
                    },
                    symbol_types=symbol_types)

            @property
            def constraint_symbols(self):
                return ['G']

            def check_constraints(self, constraint_inputs):
                return constraint_inputs['G'] == 0

        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(symbols)
        models['model4'] = Model4(symbol_types=symbols)
        material = GraphTest.generate_canonical_material(symbols)
        g = Graph(materials=[material], symbol_types=symbols, models=models)
        g.evaluate()

        expected_quantities = [
            Quantity(symbols['A'], 19, {material}),
            Quantity(symbols['A'], 23, {material}),
            Quantity(symbols['B'], 38, {material}),
            Quantity(symbols['B'], 46, {material}),
            Quantity(symbols['C'], 57, {material}),
            Quantity(symbols['C'], 69, {material}),
            Quantity(symbols['G'], 95, {material}),
            Quantity(symbols['G'], 115, {material}),
            Quantity(symbols['F'], 266, {material}),
            Quantity(symbols['F'], 322, {material}),
            Quantity(symbols['D'], 70395, {material}),
            Quantity(symbols['D'], 85215, {material}),
            Quantity(symbols['D'], 85215, {material}),
            Quantity(symbols['D'], 103155, {material}),
            Quantity(symbols['A'], 318326190, {material}),
            Quantity(symbols['A'], 385342230, {material}),
            Quantity(symbols['A'], 385342230, {material}),
            Quantity(symbols['A'], 466466910, {material}),
            Quantity(symbols['A'], 385342230, {material}),
            Quantity(symbols['A'], 466466910, {material}),
            Quantity(symbols['A'], 466466910, {material}),
            Quantity(symbols['A'], 564670470, {material})
        ]

        for q in expected_quantities:
            self.assertTrue(q in g._symbol_to_quantity[q.symbol],
                            "Evaluate failed to derive all outputs.")
            self.assertTrue(material in q._material,
                            "Evaluate failed to assign material.")

    def test_evaluate_single_material_degenerate_property(self):
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
        propnet = Graph()
        mat1 = Material()
        mat1.add_quantity(Quantity(DEFAULT_SYMBOLS['relative_permeability'], 1, None))
        mat1.add_quantity(Quantity(DEFAULT_SYMBOLS['relative_permeability'], 2, None))
        mat1.add_quantity(Quantity(DEFAULT_SYMBOLS['relative_permittivity'], 3, None))
        mat1.add_quantity(Quantity(DEFAULT_SYMBOLS['relative_permittivity'], 5, None))
        propnet.add_material(mat1)

        propnet.evaluate(material=mat1)

        # Expected outputs
        s_outputs = []
        s_outputs.append(Quantity('relative_permeability', 1, None))
        s_outputs.append(Quantity('relative_permeability', 2, None))
        s_outputs.append(Quantity('relative_permittivity', 3, None))
        s_outputs.append(Quantity('relative_permittivity', 5, None))
        s_outputs.append(Quantity('refractive_index', 3 ** 0.5, None))
        s_outputs.append(Quantity('refractive_index', 5 ** 0.5, None))
        s_outputs.append(Quantity('refractive_index', 6 ** 0.5, None))
        s_outputs.append(Quantity('refractive_index', 10 ** 0.5, None))

        m_outputs = [mat1]

        st_outputs = []
        st_outputs.append(DEFAULT_SYMBOLS['relative_permeability'])
        st_outputs.append(DEFAULT_SYMBOLS['relative_permittivity'])
        st_outputs.append(DEFAULT_SYMBOLS['refractive_index'])

        # Test
        for q_expected in s_outputs:
            q = None
            for q_derived in propnet._symbol_to_quantity[q_expected.symbol]:
                if q_derived == q_expected:
                    q = q_derived
            self.assertTrue(q is not None,
                            "Quantity missing from evaluate.")
            self.assertTrue(mat1 in q._material,
                                "Incorrect material assignment.")

    def test_evaluate_double_material_non_degenerate_property_1(self):
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
        propnet = Graph()
        mat1 = Material()
        mat2 = Material()
        mat1.add_quantity(Quantity('relative_permeability', 1, None))
        mat2.add_quantity(Quantity('relative_permeability', 2, None))
        mat1.add_quantity(Quantity('relative_permittivity', 3, None))
        mat2.add_quantity(Quantity('relative_permittivity', 5, None))
        propnet.add_material(mat1)
        propnet.add_material(mat2)

        propnet.evaluate()

        # Expected propnet outputs
        s_outputs = []
        s_outputs.append(Quantity('relative_permeability', 1, None))
        s_outputs.append(Quantity('relative_permeability', 2, None))
        s_outputs.append(Quantity('relative_permittivity', 3, None))
        s_outputs.append(Quantity('relative_permittivity', 5, None))
        s_outputs.append(Quantity('refractive_index', 3 ** 0.5, None))
        s_outputs.append(Quantity('refractive_index', 5 ** 0.5, None))
        s_outputs.append(Quantity('refractive_index', 6 ** 0.5, None))
        s_outputs.append(Quantity('refractive_index', 10 ** 0.5, None))

        m_outputs = [mat1, mat2]

        # Expected mat_1 outputs
        m1_s_outputs = []
        m1_s_outputs.append(Quantity('relative_permeability', 1, None))
        m1_s_outputs.append(Quantity('relative_permittivity', 3, None))
        m1_s_outputs.append(Quantity('refractive_index', 3 ** 0.5, None))

        #Expected mat_2 outputs
        m2_s_outputs = []
        m2_s_outputs.append(Quantity('relative_permeability', 2, None))
        m2_s_outputs.append(Quantity('relative_permittivity', 5, None))
        m2_s_outputs.append(Quantity('refractive_index', 10 ** 0.5, None))

        st_outputs = []
        st_outputs.append(DEFAULT_SYMBOLS['relative_permeability'])
        st_outputs.append(DEFAULT_SYMBOLS['relative_permittivity'])
        st_outputs.append(DEFAULT_SYMBOLS['refractive_index'])

        # Test
        for q_expected in s_outputs:
            q = None
            for q_derived in propnet._symbol_to_quantity[q_expected.symbol]:
                if q_derived == q_expected:
                    q = q_derived
            self.assertTrue(q is not None,
                            "Quantity missing from evaluate.")
            if q in m1_s_outputs:
                self.assertTrue(mat1 in q._material and mat2 not in q._material,
                                "Incorrect material assignment.")
            elif q in m2_s_outputs:
                self.assertTrue(mat2 in q._material and mat1 not in q._material,
                                "Incorrect material assignment.")
            else:
                self.assertTrue(mat1 in q._material and mat2 in q._material,
                                "Incorrect material assignment.")

    def test_evaluate_double_material_non_degenerate_property_2(self):
        """
        Graph has two materials on it mat1 and mat2.
            mat1 has nondegenerate properties A=2 & B=3
            mat2 has nondegenerate properties B=5 & C=7

        Graph has two models on it:
            model1 takes in properties A & B to produce property C=A*B
            model2 takes in properties C & B to produce property G=C/B

        We expect propagate to create a graph structure as follows:
            mat1 has properties: C=6, G=2
            mat2 has properties: G=7/5
            Joint properties: G=6/5, G=7/3
        """

        # Setup

        a = Symbol('a', ['A'], ['A'], units=[1.0, []], shape=[1])
        b = Symbol('b', ['A'], ['A'], units=[1.0, []], shape=[1])
        c = Symbol('c', ['A'], ['A'], units=[1.0, []], shape=[1])
        d = Symbol('d', ['A'], ['A'], units=[1.0, []], shape=[1])
        symbol_type_dict = {'a': a, 'b': b, 'c': c, 'd': d}

        mat1 = Material()
        mat2 = Material()
        mat1.add_quantity(Quantity(a, 2, []))
        mat1.add_quantity(Quantity(b, 3, []))
        mat2.add_quantity(Quantity(b, 5, []))
        mat2.add_quantity(Quantity(c, 7, []))

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
                        'equations': ['c-a*b'],
                        'description': ''
                    },
                                       symbol_types=symbol_types)

        class Model2 (AbstractModel):
            def __init__(self, symbol_types=None):
                AbstractModel.__init__(self, metadata={
                        'title': 'model2',
                        'tags': [],
                        'references': [],
                        'symbol_mapping': {'d': 'd',
                                           'c': 'c',
                                           'b': 'b'},
                        'connections': [{'inputs': ['c', 'b'],
                                         'outputs': ['d']
                                         }],
                        'equations': ['d-c*b'],
                        'description': ''
                    },
                                       symbol_types=symbol_types)

        p = Graph(materials=[mat1, mat2],
                  models={'model1': Model1(symbol_types=symbol_type_dict),
                          'model2': Model2(symbol_types=symbol_type_dict)},
                  symbol_types=symbol_type_dict)

        # Evaluate
        p.evaluate()

        # Test
        m1_s_outputs = []
        m1_s_outputs.append(Quantity(a, 2, []))
        m1_s_outputs.append(Quantity(b, 3, []))
        m1_s_outputs.append(Quantity(c, 6, []))
        m1_s_outputs.append(Quantity(d, 18, []))

        m2_s_outputs = []
        m2_s_outputs.append(Quantity(b, 5, []))
        m2_s_outputs.append(Quantity(c, 7, []))
        m2_s_outputs.append(Quantity(d, 35, []))

        joint_outputs = []
        joint_outputs.append(Quantity(d, 30, []))
        joint_outputs.append(Quantity(d, 21, []))
        joint_outputs.append(Quantity(d, 50, []))

        for q_expected in m1_s_outputs:
            q = None
            for q_derived in p._symbol_to_quantity[q_expected.symbol]:
                if q_derived == q_expected:
                    q = q_derived
            self.assertTrue(q is not None,
                            "Quantity missing from evaluate.")
            self.assertTrue(mat1 in q._material,
                            "Evaluate incorrectly assigned material.")
            self.assertTrue(mat2 not in q._material,
                            "Evaluate incorrectly assigned material.")

        for q_expected in m2_s_outputs:
            q = None
            for q_derived in p._symbol_to_quantity[q_expected.symbol]:
                if q_derived == q_expected:
                    q = q_derived
            self.assertTrue(q is not None,
                            "Quantity missing from evaluate.")
            self.assertTrue(mat2 in q._material,
                            "Evaluate incorrectly assigned material.")
            self.assertTrue(mat1 not in q._material,
                            "Evaluate incorrectly assigned material.")

        for q_expected in joint_outputs:
            q = None
            for q_derived in p._symbol_to_quantity[q_expected.symbol]:
                if q_derived == q_expected:
                    q = q_derived
            self.assertTrue(q is not None,
                            "Quantity missing from evaluate.")
            self.assertTrue(mat1 in q._material,
                            "Evaluate incorrectly assigned material.")
            self.assertTrue(mat2 in q._material,
                            "Evaluate incorrectly assigned material.")

    def test_evaluate_with_constraint(self):
        """
        Simple graph in which property C can be derived from properties A and B iff Constraint has value True.
        Graph has 2 materials on it: mat1 and mat2.
            mat1 has a Constraint with value True.
            mat2 has a Constraint with value False.
        Upon evaluation of the graph, we expect c to be derived ONLY for mat1.
        """

        # Setup

        a = Symbol('a', ['A'], ['A'], units=[1.0, []], shape=[1])
        b = Symbol('b', ['A'], ['A'], units=[1.0, []], shape=[1])
        c = Symbol('c', ['A'], ['A'], units=[1.0, []], shape=[1])
        constraint = Symbol('constraint', ['C'], ['C'], category='object')
        symbol_type_dict = {'a': a, 'b': b, 'c': c, 'constraint': constraint}

        a_example = Quantity(a, 2, [])
        b_example = Quantity(b, 3, [])
        const1 = Quantity(constraint, True, [])
        const2 = Quantity(constraint, False, [])

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
                        'equations': ['c-a*b'],
                        'description': ''
                    },
                                       symbol_types=symbol_types)

            @property
            def constraint_symbols(self):
                return ['const']

            def check_constraints(self, ins):
                return ins['const']

        mat1 = Material()
        mat1.add_quantity(a_example)
        mat1.add_quantity(b_example)
        mat1.add_quantity(const1)

        mat2 = Material()
        mat2.add_quantity(a_example)
        mat2.add_quantity(b_example)
        mat2.add_quantity(const2)

        p = Graph(materials=[mat1, mat2],
                  models={'model1': Model1(symbol_type_dict)},
                  symbol_types=symbol_type_dict)

        # Evaluate
        p.evaluate(material=mat1)
        p.evaluate(material=mat2)

        # Test
        m1_s_outputs = []
        m1_s_outputs.append(Quantity(a, 2, []))
        m1_s_outputs.append(Quantity(b, 3, []))
        m1_s_outputs.append(Quantity(c, 6, []))
        m1_s_outputs.append(Quantity(constraint, True, []))

        m2_s_outputs = []
        m2_s_outputs.append(Quantity(a, 2, []))
        m2_s_outputs.append(Quantity(b, 3, []))
        m2_s_outputs.append(Quantity(constraint, False, []))

        for q_expected in m1_s_outputs:
            q = None
            for q_derived in p._symbol_to_quantity[q_expected.symbol]:
                if q_derived == q_expected:
                    q = q_derived
            self.assertTrue(q is not None,
                            "Quantity missing from evaluate.")

        for q_expected in m2_s_outputs:
            q = None
            for q_derived in p._symbol_to_quantity[q_expected.symbol]:
                if q_derived == q_expected:
                    q = q_derived
            self.assertTrue(q is not None,
                            "Quantity missing from evaluate.")

    def test_symbol_expansion(self):
        """
        Tests the Symbol Expansion algorithm on a non-cyclic graph.
        The canonical graph and the canonical material are used for this test.
        """
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(symbols)
        material = GraphTest.generate_canonical_material(symbols)
        del models['model6']
        g = Graph(materials=[material], symbol_types=symbols, models=models)

        ts = []
        ans = []

        ts.append(g.calculable_properties({symbols['A']}))
        ans.append({x for x in symbols.values() if x is not symbols['A']})

        ts.append(g.calculable_properties({symbols['B']}))
        ans.append({symbols['F']})

        ts.append(g.calculable_properties({symbols['C']}))
        ans.append(set())

        ts.append(g.calculable_properties({symbols['C'], symbols['G']}))
        ans.append({symbols['D']})

        ts.append(g.calculable_properties({symbols['B'], symbols['C']}))
        ans.append({symbols['D'], symbols['F']})

        for i in range(0, len(ts)):
            self.assertTrue(ts[i] == ans[i],
                            "Symbol Expansion failed: test - " + str(i))

    def test_symbol_expansion_cyclic(self):
        """
        Tests the Symbol Expansion algorithm on a cyclic graph.
        The canonical graph and the canonical material are used for this test.
        """
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(symbols)
        material = GraphTest.generate_canonical_material(symbols)
        g = Graph(materials=[material], symbol_types=symbols, models=models)

        ts = []
        ans = []

        ts.append(g.calculable_properties({symbols['A']}))
        ans.append({x for x in symbols.values() if x is not symbols['A']})

        ts.append(g.calculable_properties({symbols['B']}))
        ans.append({symbols['F']})

        ts.append(g.calculable_properties({symbols['C']}))
        ans.append(set())

        ts.append(g.calculable_properties({symbols['C'], symbols['G']}))
        ans.append({symbols['D']})

        ts.append(g.calculable_properties({symbols['B'], symbols['C']}))
        ans.append({x for x in symbols.values() if x is not symbols['B'] and x is not symbols['C']})

        for i in range(0, len(ts)):
            self.assertTrue(ts[i] == ans[i],
                            "Symbol Expansion failed: test - " + str(i))

    def test_symbol_expansion_constraints(self):
        """
        Tests the Symbol Expansion algorithm on a non-cyclic graph with constraints.
        The canonical graph and the canonical material are used for this test.
        """
        class Model4 (AbstractModel):
            def __init__(self, symbol_types=None):
                AbstractModel.__init__(self, metadata=
                    { 'title': 'model4', 'tags': [], 'references': [], 'description': '',
                      'symbol_mapping': {'D': 'D',
                                         'B': 'B',
                                         'C': 'C',
                                         'G': 'G'
                                        },
                      'connections': [{'inputs': ['B', 'C'],
                                       'outputs': ['D']
                                      }],
                      'equations': ['D-B*C*11']
                    },
                    symbol_types=symbol_types)

            @property
            def constraint_symbols(self):
                return ['G']

            def check_constraints(self, constraint_inputs):
                return constraint_inputs['G'] == 0

        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(symbols)
        models['model4'] = Model4(symbol_types=symbols)
        del models['model6']
        material = GraphTest.generate_canonical_material(symbols)
        g = Graph(materials=[material], symbol_types=symbols, models=models)

        ts = []
        ans = []

        ts.append(g.calculable_properties({symbols['A']}))
        ans.append({x for x in symbols.values() if x is not symbols['A']})

        ts.append(g.calculable_properties({symbols['B']}))
        ans.append({symbols['F']})

        ts.append(g.calculable_properties({symbols['C']}))
        ans.append(set())

        ts.append(g.calculable_properties({symbols['C'], symbols['G']}))
        ans.append({symbols['D']})

        ts.append(g.calculable_properties({symbols['B'], symbols['C']}))
        ans.append({symbols['F']})

        for i in range(0, len(ts)):
            self.assertEqual(ts[i], ans[i],
                             "Symbol Expansion failed: test - " + str(i))

    def test_symbol_expansion_cyclic_constraints(self):
        """
        Tests the Symbol Expansion algorithm on a cyclic graph with constraints.
        The canonical graph and the canonical material are used for this test.
        """
        class Model4 (AbstractModel):
            def __init__(self, symbol_types=None):
                AbstractModel.__init__(self, metadata=
                    { 'title': 'model4', 'tags': [], 'references': [], 'description': '',
                      'symbol_mapping': {'D': 'D',
                                         'B': 'B',
                                         'C': 'C',
                                         'G': 'G'
                                        },
                      'connections': [{'inputs': ['B', 'C'],
                                       'outputs': ['D']
                                      }],
                      'equations': ['D-B*C*11']
                    },
                    symbol_types=symbol_types)

            @property
            def constraint_symbols(self):
                return ['G']

            def check_constraints(self, constraint_inputs):
                return constraint_inputs['G'] == 0

        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(symbols)
        models['model4'] = Model4(symbol_types=symbols)
        material = GraphTest.generate_canonical_material(symbols)
        g = Graph(materials=[material], symbol_types=symbols, models=models)

        ts = []
        ans = []

        ts.append(g.calculable_properties({symbols['A']}))
        ans.append({x for x in symbols.values() if x is not symbols['A']})

        ts.append(g.calculable_properties({symbols['B']}))
        ans.append({symbols['F']})

        ts.append(g.calculable_properties({symbols['C']}))
        ans.append(set())

        ts.append(g.calculable_properties({symbols['C'], symbols['G']}))
        ans.append({symbols['D']})

        ts.append(g.calculable_properties({symbols['B'], symbols['C']}))
        ans.append({symbols['F']})

        for i in range(0, len(ts)):
            self.assertEqual(ts[i], ans[i],
                             "Symbol Expansion failed: test - " + str(i))

    def test_symbol_ancestry(self):
        """
        Tests the Symbol Ancestry algorithm on a non-cyclic graph.
        The canonical graph and the canonical material are used for this test.
        """
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(symbols)
        material = GraphTest.generate_canonical_material(symbols)
        del models['model6']
        g = Graph(materials=[material], symbol_types=symbols, models=models)

        out1 = g.required_inputs_for_property(symbols['F'])

        self.assertTrue(out1.head.m is None and out1.head.parent is None and
                        out1.head.inputs == {symbols['F']} and len(out1.head.children) == 1,
                        "Tree head not properly defined.")
        self.assertTrue(out1.head.children[0].m == models['model3'] and
                        out1.head.children[0].inputs == {symbols['B']} and
                        out1.head.children[0].parent is out1.head and
                        len(out1.head.children[0].children) == 1,
                        "Tree branch improperly formed.")
        self.assertTrue(out1.head.children[0].children[0].m == models['model1'] and
                        out1.head.children[0].children[0].inputs == {symbols['A']} and
                        out1.head.children[0].children[0].parent is out1.head.children[0] and
                        len(out1.head.children[0].children) == 1 and
                        len(out1.head.children[0].children[0].children) == 0,
                        "Tree branch improperly formed.")

        out2 = g.required_inputs_for_property(symbols['D'])
        self.assertTrue(out2.head.m is None and out2.head.parent is None and
                        out2.head.inputs == {symbols['D']} and len(out2.head.children) == 2,
                        "Tree head not properly defined.")
        m_map = {x.m: x for x in out2.head.children}
        self.assertTrue(m_map[models['model4']].inputs == {symbols['B'], symbols['C']} and
                        m_map[models['model4']].parent is out2.head,
                        "Tree branch improperly formed.")
        self.assertTrue(m_map[models['model5']].inputs == {symbols['C'], symbols['G']} and
                        m_map[models['model5']].parent is out2.head and
                        len(m_map[models['model5']].children) == 2,
                        "Tree branch improperly formed.")
        m_map_1 = {x.m: x for x in m_map[models['model4']].children}
        m_map_2 = {x.m: x for x in m_map[models['model5']].children}
        self.assertTrue(m_map_1[models['model1']].inputs == {symbols['A']} and
                        m_map_1[models['model1']].parent is m_map[models['model4']] and
                        m_map_1[models['model1']].children == [],
                        "Tree branch improperly formed.")
        self.assertTrue(m_map_2[models['model1']].inputs == {symbols['G'], symbols['A']} and
                        m_map_2[models['model1']].parent is m_map[models['model5']] and
                        len(m_map_2[models['model1']].children) == 1 and
                        m_map_2[models['model1']].children[0].parent is m_map_2[models['model1']] and
                        m_map_2[models['model1']].children[0].children == [] and
                        m_map_2[models['model1']].children[0].inputs == {symbols['A']},
                        "Tree branch improperly formed.")
        self.assertTrue(m_map_2[models['model2']].inputs == {symbols['C'], symbols['A']} and
                        m_map_2[models['model2']].parent is m_map[models['model5']] and
                        len(m_map_2[models['model2']].children) == 1 and
                        m_map_2[models['model2']].children[0].parent is m_map_2[models['model2']] and
                        m_map_2[models['model2']].children[0].children == [] and
                        m_map_2[models['model2']].children[0].inputs == {symbols['A']},
                        "Tree branch improperly formed.")

    def test_symbol_ancestry_cyclic(self):
        """
        Tests the Symbol Ancestry algorithm on a cyclic graph.
        The canonical graph and the canonical material are used for this test.
        """
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(symbols)
        material = GraphTest.generate_canonical_material(symbols)
        g = Graph(materials=[material], symbol_types=symbols, models=models)

        out1 = g.required_inputs_for_property(symbols['F'])

        self.assertTrue(out1.head.m is None and out1.head.parent is None and
                        out1.head.inputs == {symbols['F']} and len(out1.head.children) == 1,
                        "Tree head not properly defined.")
        self.assertTrue(out1.head.children[0].m == models['model3'] and
                        out1.head.children[0].inputs == {symbols['B']} and
                        out1.head.children[0].parent is out1.head and
                        len(out1.head.children[0].children) == 1,
                        "Tree branch improperly formed.")
        self.assertTrue(out1.head.children[0].children[0].m == models['model1'] and
                        out1.head.children[0].children[0].inputs == {symbols['A']} and
                        out1.head.children[0].children[0].parent is out1.head.children[0] and
                        len(out1.head.children[0].children) == 1 and
                        len(out1.head.children[0].children[0].children) == 0,
                        "Tree branch improperly formed.")

        out2 = g.required_inputs_for_property(symbols['D'])
        self.assertTrue(out2.head.m is None and out2.head.parent is None and
                        out2.head.inputs == {symbols['D']} and len(out2.head.children) == 2,
                        "Tree head not properly defined.")
        m_map = {x.m: x for x in out2.head.children}
        self.assertTrue(m_map[models['model4']].inputs == {symbols['B'], symbols['C']} and
                        m_map[models['model4']].parent is out2.head,
                        "Tree branch improperly formed.")
        self.assertTrue(m_map[models['model5']].inputs == {symbols['C'], symbols['G']} and
                        m_map[models['model5']].parent is out2.head and
                        len(m_map[models['model5']].children) == 2,
                        "Tree branch improperly formed.")
        m_map_1 = {x.m: x for x in m_map[models['model4']].children}
        m_map_2 = {x.m: x for x in m_map[models['model5']].children}
        self.assertTrue(m_map_1[models['model1']].inputs == {symbols['A']} and
                        m_map_1[models['model1']].parent is m_map[models['model4']] and
                        m_map_1[models['model1']].children == [],
                        "Tree branch improperly formed.")
        self.assertTrue(m_map_2[models['model1']].inputs == {symbols['G'], symbols['A']} and
                        m_map_2[models['model1']].parent is m_map[models['model5']] and
                        len(m_map_2[models['model1']].children) == 1 and
                        m_map_2[models['model1']].children[0].parent is m_map_2[models['model1']] and
                        m_map_2[models['model1']].children[0].children == [] and
                        m_map_2[models['model1']].children[0].inputs == {symbols['A']},
                        "Tree branch improperly formed.")
        self.assertTrue(m_map_2[models['model2']].inputs == {symbols['C'], symbols['A']} and
                        m_map_2[models['model2']].parent is m_map[models['model5']] and
                        len(m_map_2[models['model2']].children) == 1 and
                        m_map_2[models['model2']].children[0].parent is m_map_2[models['model2']] and
                        m_map_2[models['model2']].children[0].children == [] and
                        m_map_2[models['model2']].children[0].inputs == {symbols['A']},
                        "Tree branch improperly formed.")

    def test_symbol_ancestry_constraint(self):
        """
        Tests the Symbol Ancestry algorithm on a non-cyclic graph with constraints.
        The canonical graph and the canonical material are used for this test.
        """
        class Model4 (AbstractModel):
            def __init__(self, symbol_types=None):
                AbstractModel.__init__(self, metadata=
                    { 'title': 'model4', 'tags': [], 'references': [], 'description': '',
                      'symbol_mapping': {'D': 'D',
                                         'B': 'B',
                                         'C': 'C',
                                         'G': 'G'
                                        },
                      'connections': [{'inputs': ['B', 'C'],
                                       'outputs': ['D']
                                      }],
                      'equations': ['D-B*C*11']
                    },
                    symbol_types=symbol_types)

            @property
            def constraint_symbols(self):
                return ['G']

            def check_constraints(self, constraint_inputs):
                return constraint_inputs['G'] == 0

        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(symbols)
        models['model4'] = Model4(symbol_types=symbols)
        del models['model6']
        g = Graph(symbol_types=symbols, models=models)

        out1 = g.required_inputs_for_property(symbols['F'])

        self.assertTrue(out1.head.m is None and out1.head.parent is None and
                        out1.head.inputs == {symbols['F']} and len(out1.head.children) == 1,
                        "Tree head not properly defined.")
        self.assertTrue(out1.head.children[0].m == models['model3'] and
                        out1.head.children[0].inputs == {symbols['B']} and
                        out1.head.children[0].parent is out1.head and
                        len(out1.head.children[0].children) == 1,
                        "Tree branch improperly formed.")
        self.assertTrue(out1.head.children[0].children[0].m == models['model1'] and
                        out1.head.children[0].children[0].inputs == {symbols['A']} and
                        out1.head.children[0].children[0].parent is out1.head.children[0] and
                        len(out1.head.children[0].children) == 1 and
                        len(out1.head.children[0].children[0].children) == 0,
                        "Tree branch improperly formed.")

        out2 = g.required_inputs_for_property(symbols['D'])
        self.assertTrue(out2.head.m is None and out2.head.parent is None and
                        out2.head.inputs == {symbols['D']} and len(out2.head.children) == 2,
                        "Tree head not properly defined.")
        m_map = {x.m: x for x in out2.head.children}
        self.assertTrue(m_map[models['model4']].inputs == {symbols['B'], symbols['C'], symbols['G']} and
                        m_map[models['model4']].parent is out2.head,
                        "Tree branch improperly formed.")
        self.assertTrue(m_map[models['model5']].inputs == {symbols['C'], symbols['G']} and
                        m_map[models['model5']].parent is out2.head and
                        len(m_map[models['model5']].children) == 2,
                        "Tree branch improperly formed.")
        m_map_2 = {x.m: x for x in m_map[models['model5']].children}
        self.assertTrue(m_map_2[models['model1']].inputs == {symbols['G'], symbols['A']} and
                        m_map_2[models['model1']].parent is m_map[models['model5']] and
                        len(m_map_2[models['model1']].children) == 1 and
                        m_map_2[models['model1']].children[0].parent is m_map_2[models['model1']] and
                        m_map_2[models['model1']].children[0].children == [] and
                        m_map_2[models['model1']].children[0].inputs == {symbols['A']},
                        "Tree branch improperly formed.")
        self.assertTrue(m_map_2[models['model2']].inputs == {symbols['C'], symbols['A']} and
                        m_map_2[models['model2']].parent is m_map[models['model5']] and
                        len(m_map_2[models['model2']].children) == 1 and
                        m_map_2[models['model2']].children[0].parent is m_map_2[models['model2']] and
                        m_map_2[models['model2']].children[0].children == [] and
                        m_map_2[models['model2']].children[0].inputs == {symbols['A']},
                        "Tree branch improperly formed.")

        m_map_1 = {x.m: x for x in m_map[models['model4']].children}
        self.assertTrue(m_map_1[models['model1']].inputs == {symbols['G'], symbols['A']} and
                        m_map_1[models['model1']].parent is m_map[models['model4']] and
                        len(m_map_1[models['model1']].children) == 1 and
                        m_map_1[models['model1']].children[0].parent is m_map_1[models['model1']] and
                        m_map_1[models['model1']].children[0].children == [] and
                        m_map_1[models['model1']].children[0].inputs == {symbols['A']},
                        "Tree branch improperly formed.")
        self.assertTrue(m_map_1[models['model2']].inputs == {symbols['B'], symbols['C'], symbols['A']} and
                        m_map_1[models['model2']].parent is m_map[models['model4']] and
                        len(m_map_1[models['model2']].children) == 1 and
                        m_map_1[models['model2']].children[0].parent is m_map_1[models['model2']] and
                        m_map_1[models['model2']].children[0].children == [] and
                        m_map_1[models['model2']].children[0].inputs == {symbols['A']},
                        "Tree branch improperly formed.")

    def test_symbol_ancestry_cyclic_constraint(self):
        """
        Tests the Symbol Ancestry algorithm on a cyclic graph with constraints.
        The canonical graph and the canonical material are used for this test.
        """
        class Model4 (AbstractModel):
            def __init__(self, symbol_types=None):
                AbstractModel.__init__(self, metadata=
                    { 'title': 'model4', 'tags': [], 'references': [], 'description': '',
                      'symbol_mapping': {'D': 'D',
                                         'B': 'B',
                                         'C': 'C',
                                         'G': 'G'
                                        },
                      'connections': [{'inputs': ['B', 'C'],
                                       'outputs': ['D']
                                      }],
                      'equations': ['D-B*C*11']
                    },
                    symbol_types=symbol_types)

            @property
            def constraint_symbols(self):
                return ['G']

            def check_constraints(self, constraint_inputs):
                return constraint_inputs['G'] == 0

        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(symbols)
        models['model4'] = Model4(symbol_types=symbols)
        g = Graph(symbol_types=symbols, models=models)

        out1 = g.required_inputs_for_property(symbols['F'])

        self.assertTrue(out1.head.m is None and out1.head.parent is None and
                        out1.head.inputs == {symbols['F']} and len(out1.head.children) == 1,
                        "Tree head not properly defined.")
        self.assertTrue(out1.head.children[0].m == models['model3'] and
                        out1.head.children[0].inputs == {symbols['B']} and
                        out1.head.children[0].parent is out1.head and
                        len(out1.head.children[0].children) == 1,
                        "Tree branch improperly formed.")
        self.assertTrue(out1.head.children[0].children[0].m == models['model1'] and
                        out1.head.children[0].children[0].inputs == {symbols['A']} and
                        out1.head.children[0].children[0].parent is out1.head.children[0] and
                        len(out1.head.children[0].children) == 1 and
                        len(out1.head.children[0].children[0].children) == 0,
                        "Tree branch improperly formed.")

        out2 = g.required_inputs_for_property(symbols['D'])
        self.assertTrue(out2.head.m is None and out2.head.parent is None and
                        out2.head.inputs == {symbols['D']} and len(out2.head.children) == 2,
                        "Tree head not properly defined.")
        m_map = {x.m: x for x in out2.head.children}
        self.assertTrue(m_map[models['model4']].inputs == {symbols['B'], symbols['C'], symbols['G']} and
                        m_map[models['model4']].parent is out2.head,
                        "Tree branch improperly formed.")
        self.assertTrue(m_map[models['model5']].inputs == {symbols['C'], symbols['G']} and
                        m_map[models['model5']].parent is out2.head and
                        len(m_map[models['model5']].children) == 2,
                        "Tree branch improperly formed.")
        m_map_2 = {x.m: x for x in m_map[models['model5']].children}
        self.assertTrue(m_map_2[models['model1']].inputs == {symbols['G'], symbols['A']} and
                        m_map_2[models['model1']].parent is m_map[models['model5']] and
                        len(m_map_2[models['model1']].children) == 1 and
                        m_map_2[models['model1']].children[0].parent is m_map_2[models['model1']] and
                        m_map_2[models['model1']].children[0].children == [] and
                        m_map_2[models['model1']].children[0].inputs == {symbols['A']},
                        "Tree branch improperly formed.")
        self.assertTrue(m_map_2[models['model2']].inputs == {symbols['C'], symbols['A']} and
                        m_map_2[models['model2']].parent is m_map[models['model5']] and
                        len(m_map_2[models['model2']].children) == 1 and
                        m_map_2[models['model2']].children[0].parent is m_map_2[models['model2']] and
                        m_map_2[models['model2']].children[0].children == [] and
                        m_map_2[models['model2']].children[0].inputs == {symbols['A']},
                        "Tree branch improperly formed.")

        m_map_1 = {x.m: x for x in m_map[models['model4']].children}
        self.assertTrue(m_map_1[models['model1']].inputs == {symbols['G'], symbols['A']} and
                        m_map_1[models['model1']].parent is m_map[models['model4']] and
                        len(m_map_1[models['model1']].children) == 1 and
                        m_map_1[models['model1']].children[0].parent is m_map_1[models['model1']] and
                        m_map_1[models['model1']].children[0].children == [] and
                        m_map_1[models['model1']].children[0].inputs == {symbols['A']},
                        "Tree branch improperly formed.")
        self.assertTrue(m_map_1[models['model2']].inputs == {symbols['B'], symbols['C'], symbols['A']} and
                        m_map_1[models['model2']].parent is m_map[models['model4']] and
                        len(m_map_1[models['model2']].children) == 1 and
                        m_map_1[models['model2']].children[0].parent is m_map_1[models['model2']] and
                        m_map_1[models['model2']].children[0].children == [] and
                        m_map_1[models['model2']].children[0].inputs == {symbols['A']},
                        "Tree branch improperly formed.")

    def test_get_path(self):
        """
        Tests the ability to generate all paths from one symbol to another.
        """
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(symbols)
        material = GraphTest.generate_canonical_material(symbols)
        del models['model6']
        g = Graph(materials=[material], symbol_types=symbols, models=models)

        paths_1 = g.get_paths(symbols['A'], symbols['F'])
        paths_2 = g.get_paths(symbols['A'], symbols['D'])

        ans_1 = [SymbolPath({symbols['A']}, [models['model1'], models['model3']])]
        ans_2 = [
                    SymbolPath({symbols['A'], symbols['C']}, [models['model2'], models['model5']]),
                    SymbolPath({symbols['A'], symbols['G']}, [models['model1'], models['model5']]),
                    SymbolPath({symbols['A']}, [models['model1'], models['model4']]),
                    SymbolPath({symbols['A']}, [models['model1'], models['model2'], models['model5']]),
                    SymbolPath({symbols['A']}, [models['model2'], models['model1'], models['model5']])
                ]
        self.assertTrue(len(paths_1) == len(ans_1),
                        "Incorrect paths generated.")
        self.assertTrue(len(paths_2) == len(ans_2),
                        "Incorrect paths generated.")
        for i in paths_1:
            self.assertTrue(i in ans_1,
                            "Incorrect paths generated.")
        for i in paths_2:
            self.assertTrue(i in ans_2,
                            "Incorrect paths generated.")
    
    def test_get_path_constraint(self):
        """
        Tests the ability to generate all paths from one symbol to another with constraints.
        """
        class Model4 (AbstractModel):
            def __init__(self, symbol_types=None):
                AbstractModel.__init__(self, metadata=
                    { 'title': 'model4', 'tags': [], 'references': [], 'description': '',
                      'symbol_mapping': {'D': 'D',
                                         'B': 'B',
                                         'C': 'C',
                                         'G': 'G'
                                        },
                      'connections': [{'inputs': ['B', 'C'],
                                       'outputs': ['D']
                                      }],
                      'equations': ['D-B*C*11']
                    },
                    symbol_types=symbol_types)

            @property
            def constraint_symbols(self):
                return ['G']

            def check_constraints(self, constraint_inputs):
                return constraint_inputs['G'] == 0

        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(symbols)
        models['model4'] = Model4(symbol_types=symbols)
        del models['model6']
        g = Graph(symbol_types=symbols, models=models)

        paths_1 = g.get_paths(symbols['A'], symbols['F'])
        paths_2 = g.get_paths(symbols['A'], symbols['D'])

        ans_1 = [SymbolPath({symbols['A']}, [models['model1'], models['model3']])]

        ans_2 = [
            SymbolPath({symbols['A'], symbols['C']}, [models['model2'], models['model5']]),
            SymbolPath({symbols['A'], symbols['G']}, [models['model1'], models['model5']]),
            SymbolPath({symbols['A'], symbols['C'], symbols['B']}, [models['model2'], models['model4']]),
            SymbolPath({symbols['A'], symbols['G']}, [models['model1'], models['model4']]),
            SymbolPath({symbols['A']}, [models['model1'], models['model2'], models['model5']]),
            SymbolPath({symbols['A']}, [models['model2'], models['model1'], models['model5']]),
            SymbolPath({symbols['A']}, [models['model1'], models['model2'], models['model4']]),
            SymbolPath({symbols['A']}, [models['model2'], models['model1'], models['model4']])
        ]

        self.assertTrue(len(paths_1) == len(ans_1),
                        "Incorrect paths generated.")
        self.assertTrue(len(paths_2) == len(ans_2),
                        "Incorrect paths generated.")
        for i in paths_1:
            self.assertTrue(i in ans_1,
                            "Incorrect paths generated.")
        for i in paths_2:
            self.assertTrue(i in ans_2,
                            "Incorrect paths generated.")
