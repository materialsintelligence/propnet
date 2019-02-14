import unittest
from propnet.core.graph import Graph
from propnet.core.provenance import SymbolPath, ProvenanceElement
from propnet.core.materials import Material
from propnet.core.materials import CompositeMaterial
from propnet.core.symbols import Symbol
from propnet.core.models import EquationModel
from propnet.core.quantity import QuantityFactory
from propnet.ext.matproj import MPRester
from propnet import ureg

import os
import json
from monty.json import MontyDecoder, jsanitize

# noinspection PyUnresolvedReferences
import propnet.symbols
from propnet.core.registry import Registry

# TODO: I think the expansion/tree traversal methods are very cool
#       and should be preserved for now, even though they don't work.e
#       We may eventually delete but I think it'd be better to try to
#       refactor as evaluate has been
NO_EXPANSION_METHODS = True
EXPANSION_METHOD_MESSAGE = "Expansion methods (TreeBuilder, etc.) are undergoing" \
                           "revision and tests are offline until complete"
# TODO: There's a lot of code duplication here that could be added to setUp

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


class GraphTest(unittest.TestCase):
    def setUp(self):
        symbols = GraphTest.generate_canonical_symbols()

        a = [QuantityFactory.create_quantity(symbols['A'], 19),
             QuantityFactory.create_quantity(symbols['A'], 23)]
        b = [QuantityFactory.create_quantity(symbols['B'], 38,
                                             provenance=ProvenanceElement(model='model1',
                                                                          inputs=[a[0]])),
             QuantityFactory.create_quantity(symbols['B'], 46,
                                             provenance=ProvenanceElement(model='model1',
                                                                          inputs=[a[1]]))]
        c = [QuantityFactory.create_quantity(symbols['C'], 57,
                                             provenance=ProvenanceElement(model='model1',
                                                                          inputs=[a[0]])),
             QuantityFactory.create_quantity(symbols['C'], 69,
                                             provenance=ProvenanceElement(model='model1',
                                                                          inputs=[a[1]]))]
        g = [QuantityFactory.create_quantity(symbols['G'], 95,
                                             provenance=ProvenanceElement(model='model2',
                                                                          inputs=[a[0]])),
             QuantityFactory.create_quantity(symbols['G'], 115,
                                             provenance=ProvenanceElement(model='model2',
                                                                          inputs=[a[1]]))]
        f = [QuantityFactory.create_quantity(symbols['F'], 266,
                                             provenance=ProvenanceElement(model='model3',
                                                                          inputs=[b[0]])),
             QuantityFactory.create_quantity(symbols['F'], 322,
                                             provenance=ProvenanceElement(model='model3',
                                                                          inputs=[b[1]]))]
        d_model4 = [QuantityFactory.create_quantity(symbols['D'], 23826,
                                             provenance=ProvenanceElement(model='model4',
                                                                          inputs=[b[0], c[0]])),
             QuantityFactory.create_quantity(symbols['D'], 28842,
                                             provenance=ProvenanceElement(model='model4',
                                                                          inputs=[b[0], c[1]])),
             QuantityFactory.create_quantity(symbols['D'], 28842,
                                             provenance=ProvenanceElement(model='model4',
                                                                          inputs=[b[1], c[0]])),
             QuantityFactory.create_quantity(symbols['D'], 34914,
                                             provenance=ProvenanceElement(model='model4',
                                                                          inputs=[b[1], c[1]]))]

        d_model5 = [QuantityFactory.create_quantity(symbols['D'], 70395,
                                             provenance=ProvenanceElement(model='model5',
                                                                          inputs=[c[0], g[0]])),
             QuantityFactory.create_quantity(symbols['D'], 85215,
                                             provenance=ProvenanceElement(model='model5',
                                                                          inputs=[c[0], g[1]])),
             QuantityFactory.create_quantity(symbols['D'], 85215,
                                             provenance=ProvenanceElement(model='model5',
                                                                          inputs=[c[1], g[0]])),
             QuantityFactory.create_quantity(symbols['D'], 103155,
                                             provenance=ProvenanceElement(model='model5',
                                                                          inputs=[c[1], g[1]]))]
        self.expected_quantities = a + b + c + d_model4 + d_model5 + f + g
        self.expected_constrained_quantities = a + b + c + d_model5 + f + g

    @staticmethod
    def generate_canonical_symbols():
        """
        Returns a set of Symbol objects used in testing.
        Returns: (dict<str, Symbol>)
        """
        A = Symbol('A', ['A'], ['A'], units="dimensionless", shape=[1])
        B = Symbol('B', ['B'], ['B'], units="dimensionless", shape=[1])
        C = Symbol('C', ['C'], ['C'], units="dimensionless", shape=[1])
        D = Symbol('D', ['D'], ['D'], units="dimensionless", shape=[1])
        G = Symbol('G', ['G'], ['G'], units="dimensionless", shape=[1])
        F = Symbol('F', ['F'], ['F'], units="dimensionless", shape=[1])
        return {
            'A': A,
            'B': B,
            'C': C,
            'D': D,
            'G': G,
            'F': F
        }

    @staticmethod
    def generate_canonical_models(constrain_model_4=False):
        """
        Returns a set of Model objects used in testing.
        Returns: (dict<str, Model>)
        """
        sym_map = GraphTest.generate_canonical_symbols()
        # TODO: Resolve the connections issue here
        model1 = EquationModel(name="model1", equations=['B=2*A', 'C=3*A'],
                               connections=[{"inputs": ["A"], "outputs": ['B', 'C']}],
                               symbol_property_map=sym_map)
        model2 = EquationModel(name="model2", equations=['G=5*A'],
                               symbol_property_map=sym_map)
        model3 = EquationModel(name="model3", equations=['F=7*B'],
                               symbol_property_map=sym_map)
        model5 = EquationModel(name="model5", equations=['D=C*G*13'],
                               symbol_property_map=sym_map)
        model6 = EquationModel(name="model6", equations=['A=F*D*17'],
                               symbol_property_map=sym_map)

        if constrain_model_4:
            model4 = EquationModel(name="model4", equations=['D=B*C*11'],
                                   constraints=["G==0"], symbol_property_map=sym_map)
        else:
            model4 = EquationModel(name="model4", equations=['D=B*C*11'],
                                   symbol_property_map=sym_map)

        models = [model1, model2, model3, model4, model5, model6]
        return {x.name : x for x in models}

    @staticmethod
    def generate_canonical_material(c_symbols):
        """
        Generates a Material with appropriately attached Quantities.
        Args:
            c_symbols: (dict<str, Symbol>) dictionary of defined materials.
        Returns:
            (Material) material with properties loaded.
        """
        q1 = QuantityFactory.create_quantity(c_symbols['A'], 19)
        q2 = QuantityFactory.create_quantity(c_symbols['A'], 23)
        m = Material()
        m.add_quantity(q1)
        m.add_quantity(q2)
        return m

    def test_graph_setup(self):
        """
        Tests the outcome of constructing the canonical graph.
        """
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models()
        g = Graph(models=models, symbol_types=symbols, composite_models=dict())
        st_c = {x for x in symbols.values()}
        st_g = g.get_symbol_types()
        m_c = {x.name: x for x in models.values()}
        m_g = g.get_models()
        self.assertTrue(st_c == st_g,
                        'Canonical constructed graph does not have the right Symbol objects.')
        self.assertTrue(m_c == m_g,
                        'Canonical constructed graph does not have the right Model objects.')
        for m in models.values():
            for input_set in m.input_sets:
                for symbol in input_set:
                    self.assertTrue(symbols[symbol] in g._input_to_model.keys(),
                                    "Canonical constructed graph does not have an edge from input: "
                                    "{} to model: {}".format(symbol, m))
                    self.assertTrue(m in g._input_to_model[symbol],
                                    "Canonical constructed graph does not have an edge from input: "
                                    "{} to model: {}".format(symbol, m))
            for output_set in m.output_sets:
                for symbol in output_set:
                    self.assertTrue(symbols[symbol] in g._output_to_model.keys(),
                                    "Canonical constructed graph does not have an edge from input: "
                                    "{} to model: {}".format(symbol, m))
                    self.assertTrue(m in g._output_to_model[symbol],
                                    "Canonical constructed graph does not have an edge from input: "
                                    "{} to model: {}".format(symbol, m))

    def test_model_add_remove(self):
        """
        Tests the outcome of adding and removing a model from the canonical graph.
        """
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models()
        g = Graph(models=models, symbol_types=symbols, composite_models=dict())
        g.remove_models({models['model6'].name: models['model6']})
        self.assertTrue(models['model6'] not in g.get_models().values(),
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
            self.assertTrue(m in g.get_models().values(),
                            "Too many models were removed.")
        g.update_models({'Model6': m6})
        self.assertTrue(m6 in g.get_models().values(),
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
        models = GraphTest.generate_canonical_models()
        g = Graph(models=models, symbol_types=symbols, composite_models=dict())
        g.remove_symbol_types({'F': symbols['F']})
        self.assertTrue(symbols['F'] not in g.get_symbol_types(),
                        "Symbol was not properly removed.")
        self.assertTrue(symbols['F'] not in g._input_to_model.keys(),
                        "Symbol was not properly removed.")
        self.assertTrue(symbols['F'] not in g._output_to_model.keys(),
                        "Symbol was not properly removed.")
        self.assertTrue(models['model3'] not in g.get_models().values(),
                        "Removing symbol did not remove a model using that symbol.")
        self.assertTrue(models['model6'] not in g.get_models().values(),
                        "Removing symbol did not remove a model using that symbol.")
        g.update_symbol_types({'F': symbols['F']})
        self.assertTrue(symbols['F'] in g.get_symbol_types(),
                       "Symbol was not properly added.")

    def test_evaluate(self):
        """
        Tests the evaluation algorithm on a non-cyclic graph.
        The canonical graph and the canonical material are used for this test.
        """
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models()
        material = GraphTest.generate_canonical_material(symbols)
        del models['model6']
        g = Graph(symbol_types=symbols, models=models, composite_models=dict())
        material_derived = g.evaluate(material)

        expected_quantities = self.expected_quantities

        self.assertTrue(material == GraphTest.generate_canonical_material(symbols),
                        "evaluate() mutated the original material argument.")

        derived_quantities = material_derived.get_quantities()
        self.assertTrue(len(expected_quantities) == len(derived_quantities),
                        "Evaluate did not correctly derive outputs.")
        for q in expected_quantities:
            self.assertTrue(q in material_derived._symbol_to_quantity[q.symbol],
                            "Evaluate failed to derive all outputs.")
            self.assertTrue(q in derived_quantities)

    def test_evaluate_cyclic(self):
        """
        Tests the evaluation algorithm on a cyclic graph.
        The canonical graph and the canonical material are used for this test.
        """
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models()
        material = GraphTest.generate_canonical_material(symbols)
        g = Graph(symbol_types=symbols, models=models, composite_models=dict())
        material_derived = g.evaluate(material)

        expected_quantities = self.expected_quantities

        self.assertTrue(material == GraphTest.generate_canonical_material(symbols),
                        "evaluate() mutated the original material argument.")

        derived_quantities = material_derived.get_quantities()
        self.assertTrue(len(expected_quantities) == len(derived_quantities),
                        "Evaluate did not correctly derive outputs.")
        for q in expected_quantities:
            self.assertTrue(q in material_derived._symbol_to_quantity[q.symbol],
                            "Evaluate failed to derive all outputs.")
            self.assertTrue(q in derived_quantities)

    def test_derive_quantities(self):
        # Simple one quantity test
        quantity = QuantityFactory.create_quantity("band_gap", 3.2)
        graph = Graph()
        qpool = graph.derive_quantities([quantity])
        new_mat = graph.evaluate(Material([quantity]))

    @unittest.skip
    def test_evaluate_constraints(self):
        """
        Tests the evaluation algorithm on a non-cyclic graph involving
        constraints.  The canonical graph and the canonical material are
        used for this test.
        """

        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(constrain_model_4=True)
        del models['model6']
        material = GraphTest.generate_canonical_material(symbols)
        g = Graph(symbol_types=symbols, models=models, composite_models=dict())
        material_derived = g.evaluate(material)

        expected_quantities = self.expected_constrained_quantities

        self.assertTrue(material == GraphTest.generate_canonical_material(symbols),
                        "evaluate() mutated the original material argument.")

        derived_quantities = material_derived.get_quantities()
        self.assertTrue(len(expected_quantities) == len(derived_quantities),
                        "Evaluate did not correctly derive outputs.")
        for q in expected_quantities:
            self.assertTrue(q in material_derived._symbol_to_quantity[q.symbol],
                            "Evaluate failed to derive all outputs.")
            self.assertTrue(q in derived_quantities)

    def test_evaluate_constraints_cyclic(self):
        """
        Tests the evaluation algorithm on a cyclic graph involving constraints.
        The canonical graph and the canonical material are used for this test.
        """
        model4 = EquationModel(name="model4", equations=["D=B*C*11"], constraints=["G==0"])

        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models()
        models['model4'] = model4
        material = GraphTest.generate_canonical_material(symbols)
        g = Graph(symbol_types=symbols, models=models, composite_models=dict())
        material_derived = g.evaluate(material)

        expected_quantities = self.expected_constrained_quantities

        self.assertTrue(material == GraphTest.generate_canonical_material(symbols),
                        "evaluate() mutated the original material argument.")

        derived_quantities = material_derived.get_quantities()
        self.assertTrue(len(expected_quantities) == len(derived_quantities),
                        "Evaluate did not correctly derive outputs.")
        for q in expected_quantities:
            self.assertTrue(q in material_derived._symbol_to_quantity[q.symbol],
                            "Evaluate failed to derive all outputs.")
            self.assertTrue(q in derived_quantities)

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
        permeability = [QuantityFactory.create_quantity(Registry("symbols")['relative_permeability'], 1),
                        QuantityFactory.create_quantity(Registry("symbols")['relative_permeability'], 2)]
        permittivity = [QuantityFactory.create_quantity(Registry("symbols")['relative_permittivity'], 3),
                        QuantityFactory.create_quantity(Registry("symbols")['relative_permittivity'], 5)]

        for q in permeability + permittivity:
            mat1.add_quantity(q)

        mat1_derived = propnet.evaluate(mat1)

        # Expected outputs
        s_outputs = permeability + permittivity + [
            QuantityFactory.create_quantity('refractive_index', 3 ** 0.5,
                                            provenance=ProvenanceElement(model="refractive_indexfrom_rel_perm",
                                                                         inputs=[permeability[0],
                                                                                 permittivity[0]])),
            QuantityFactory.create_quantity('refractive_index', 5 ** 0.5,
                                            provenance=ProvenanceElement(model="refractive_indexfrom_rel_perm",
                                                                         inputs=[permeability[0],
                                                                                 permittivity[1]])),
            QuantityFactory.create_quantity('refractive_index', 6 ** 0.5,
                                            provenance=ProvenanceElement(model="refractive_indexfrom_rel_perm",
                                                                         inputs=[permeability[1],
                                                                                 permittivity[0]])),
            QuantityFactory.create_quantity('refractive_index', 10 ** 0.5,
                                            provenance=ProvenanceElement(model="refractive_indexfrom_rel_perm",
                                                                         inputs=[permeability[1],
                                                                                 permittivity[1]]))]

        # st_outputs = [Registry("symbols")['relative_permeability'],
        #               Registry("symbols")['relative_permittivity'],
        #               Registry("symbols")['refractive_index']]

        # Test
        for q_expected in s_outputs:
            q = None
            for q_derived in mat1_derived._symbol_to_quantity[q_expected.symbol]:
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
        models = GraphTest.generate_canonical_models()
        material = GraphTest.generate_canonical_material(symbols)
        del models['model6']
        g = Graph(symbol_types=symbols, models=models, composite_models=dict())

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
                            "Symbol Expansion failed: test - " + str(i))\


    def test_symbol_expansion_cyclic(self):
        """
        Tests the Symbol Expansion algorithm on a cyclic graph.
        The canonical graph and the canonical material are used for this test.
        """
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models()
        material = GraphTest.generate_canonical_material(symbols)
        g = Graph(symbol_types=symbols, models=models, composite_models=dict())

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
        model4 = EquationModel("model4", ['D=B*C*11'], constraints=["G==0"])

        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(constrain_model_4=True)
        models['model4'] = model4
        del models['model6']
        material = GraphTest.generate_canonical_material(symbols)
        g = Graph(symbol_types=symbols, models=models, composite_models=dict())

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
        model4 = EquationModel("model4", ['D=B*C*11'], constraints=["G==0"])
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(constrain_model_4=True)
        models['model4'] = model4
        material = GraphTest.generate_canonical_material(symbols)
        g = Graph(symbol_types=symbols, models=models, composite_models=dict())

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
        models = GraphTest.generate_canonical_models()
        material = GraphTest.generate_canonical_material(symbols)
        del models['model6']
        g = Graph(symbol_types=symbols, models=models, composite_models=dict())

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
        models = GraphTest.generate_canonical_models()
        material = GraphTest.generate_canonical_material(symbols)
        g = Graph(symbol_types=symbols, models=models, composite_models=dict())

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
        model4 = EquationModel("model4", ['D=B*C*11'], constraints=["G==0"])
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(constrain_model_4=True)
        models['model4'] = model4
        del models['model6']
        g = Graph(symbol_types=symbols, models=models, composite_models=dict())

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
        model4 = EquationModel("model4", ['D=B*C*11'], constraints=["G==0"])
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(constrain_model_4=True)
        models['model4'] = model4
        g = Graph(symbol_types=symbols, models=models, composite_models=dict())

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
        models = GraphTest.generate_canonical_models()
        material = GraphTest.generate_canonical_material(symbols)
        del models['model6']
        g = Graph(symbol_types=symbols, models=models, composite_models=dict())

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
        model4 = EquationModel("model4", ['D=B*C*11'], constraints=["G==0"])
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models(constrain_model_4=True)
        models['model4'] = model4
        del models['model6']
        g = Graph(symbol_types=symbols, models=models, composite_models=dict())

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

    @unittest.skip
    def test_super_evaluate(self):
        """
        Tests the graph's composite material evaluation.
        """
        mp_data = {}
        for mpid in ('mp-13', 'mp-24972'):
            with open(os.path.join(TEST_DIR, "{}.json".format(mpid)), 'r') as f:
                data = json.load(f)
            material = Material()
            for d in data:
                q = MontyDecoder().process_decoded(d)
                material.add_quantity(q)
            mp_data[mpid] = material

        m1 = mp_data["mp-13"]
        # Temporary hack for problem with zero band-gap materials
        m1.remove_symbol("band_gap_pbe")
        m1.add_quantity(QuantityFactory.create_quantity("band_gap", 0.0))
        m2 = mp_data["mp-24972"]
        sm = CompositeMaterial([m1, m2])

        g = Graph()

        sm = g.super_evaluate(sm, allow_model_failure=False)

        self.assertTrue('pilling_bedworth_ratio' in sm._symbol_to_quantity.keys(),
                        "Super Evaluate failed to derive expected outputs.")
        self.assertTrue(len(sm._symbol_to_quantity['pilling_bedworth_ratio']) > 0,
                        "Super Evaluate failed to derive expected outputs.")

    def test_provenance(self):
        model4 = EquationModel(name="model4", equations=["D=B*C*11"], constraints=["G==0"])
        symbols = GraphTest.generate_canonical_symbols()
        models = GraphTest.generate_canonical_models()
        models['model4'] = model4
        del models['model6']
        material = GraphTest.generate_canonical_material(symbols)
        g = Graph(symbol_types=symbols, models=models, composite_models=dict())
        material_derived = g.evaluate(material)

        expected_quantities = self.expected_constrained_quantities

        for q in material_derived._symbol_to_quantity[symbols['A']]:
            self.assertTrue(q._provenance.inputs is None)
        for q in material_derived._symbol_to_quantity[symbols['B']]:
            if q.value == 38:
                self.assertTrue(q._provenance.model is models['model1'].name,
                                "provenance improperly calculated")
                self.assertTrue(expected_quantities[0] in q._provenance.inputs,
                                "provenance improperly calculated")
            else:
                self.assertTrue(q._provenance.model is models['model1'].name,
                                "provenance improperly calculated")
                self.assertTrue(expected_quantities[1] in q._provenance.inputs,
                                "provenance improperly calculated")
        for q in material_derived._symbol_to_quantity[symbols['C']]:
            if q.value == 57:
                self.assertTrue(q._provenance.model is models['model1'].name,
                                "provenance improperly calculated")
                self.assertTrue(expected_quantities[0] in q._provenance.inputs,
                                "provenance improperly calculated")
            else:
                self.assertTrue(q._provenance.model is models['model1'].name,
                                "provenance improperly calculated")
                self.assertTrue(expected_quantities[1] in q._provenance.inputs,
                                "provenance improperly calculated")
        for q in material_derived._symbol_to_quantity[symbols['G']]:
            if q.value == 95:
                self.assertTrue(q._provenance.model is models['model2'].name,
                                "provenance improperly calculated")
                self.assertTrue(expected_quantities[0] in q._provenance.inputs,
                                "provenance improperly calculated")
            else:
                self.assertTrue(q._provenance.model is models['model2'].name,
                                "provenance improperly calculated")
                self.assertTrue(expected_quantities[1] in q._provenance.inputs,
                                "provenance improperly calculated")
        for q in material_derived._symbol_to_quantity[symbols['D']]:
            if q.value == 70395:
                self.assertTrue(q._provenance.model is models['model5'].name,
                                "provenance improperly calculated")
                self.assertTrue(expected_quantities[4] in q._provenance.inputs,
                                "provenance improperly calculated")
                self.assertTrue(expected_quantities[12] in q._provenance.inputs,
                                "provenance improperly calculated")

    @unittest.skip("Skipping creating composite data files")
    def test_generate_composite_data_files(self):
        mpr = MPRester()
        mpids = ['mp-13', 'mp-24972']
        materials = mpr.get_materials_for_mpids(mpids)
        for m in materials:
            mpid = [q.value for q in m.get_quantities() if q.symbol == "external_identifier_mp"][0]
            with open(os.path.join(TEST_DIR, '{}.json'.format(mpid)), 'w') as f:
                qs = jsanitize(m.get_quantities(), strict=True)
                f.write(json.dumps(qs))

if __name__ == "__main__":
    unittest.main()

