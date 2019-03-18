import unittest
import json

from propnet.core.graph import Graph
from propnet.symbols import add_builtin_symbols_to_registry
import os

no_store_file = os.environ.get('PROPNET_STORE_FILE') is None
if not no_store_file:
    from propnet.web.app import app, symbol_layout, model_layout
    from propnet.web.utils import graph_conversion

routes = [
    '/'
]

add_builtin_symbols_to_registry()


@unittest.skipIf(no_store_file,
                 "No data store provided. Skipping web tests.")
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

    def test_property(self):
        properties = self.client.get('/property')
        self.assertEqual(properties.status_code, 200)

    def test_symbol_layout(self):
        layout = symbol_layout("applied_stress")
        self.assertEqual(layout.children[0], "Applied stress")
        # test symbol layout for single node
        layout = symbol_layout("grain_diameter")
        self.assertTrue(layout.children[0], "Average grain diameter")

    def test_model_layout(self):
        layout = model_layout("density_relations")
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
