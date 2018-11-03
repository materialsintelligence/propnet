import unittest
import json

from propnet.web.app import app, retrieve_material, symbol_layout,\
    model_layout, graph_conversion
from propnet.core.graph import Graph

routes = [
    '/',
    '/models'
    '/properties',
    '/developer',
    '/utilities',
    '/utilities/load/materials'
]

# auto add final /
class WebTest(unittest.TestCase):
    """
    Base class for dash unittests
    """
    def setUp(self):
        self.app = app
        self.client = self.app.server.test_client()

    def test_home(self):
        home = self.client.get('/')
        self.assertEqual(home.status_code, 200)

    def test_models(self):
        models = self.client.get('/models')
        self.assertEqual(models.status_code, 200)

    def test_developer(self):
        developer = self.client.get('/developer')
        self.assertEqual(developer.status_code, 200)

    def test_utilities(self):
        utilities = self.client.get('/utilities')
        self.assertEqual(utilities.status_code, 200)

    def test_property(self):
        properties = self.client.get('/property')
        self.assertEqual(properties.status_code, 200)

    def test_load_material(self):
        # Test retrieve material method
        material_no_derive = retrieve_material(1, "Ag", [])
        self.assertEqual(material_no_derive.status_code, 200)
        material_derive = retrieve_material(1, "Ag", ['derive'])
        self.assertEqual(material_derive.status_code, 200)
        material_aggregate = retrieve_material(1, "Ag", ['derive', 'aggregate'])
        self.assertEqual(material_aggregate.status_code, 200)

    def test_symbol_layout(self):
        layout = symbol_layout("applied_stress")
        self.assertEqual(layout.children[0], "Applied stress")
        # test symbol layout for single node
        layout = symbol_layout("grain_diameter")
        self.assertTrue(layout.children[0], "Average grain diameter")

    def test_model_layout(self):
        layout = model_layout("density")
        self.assertTrue(layout.children[0], "Atomic Density")

    def test_graph_conversion(self):
        graph = Graph()
        converted = graph_conversion(graph.get_networkx_graph())
        serialized = json.dumps(converted)
        self.assertIsNotNone(serialized)
        # Ensure that there are both nodes and proper edges
        self.assertIn('Band gap', [n['label'] for n in converted['nodes']])
        self.assertIn({'from': 'band_gap', "to": "Is Metallic"},
                      converted['edges'])

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
