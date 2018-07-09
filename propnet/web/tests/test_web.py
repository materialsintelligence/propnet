import unittest
from propnet.web.app import app

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
    Base class for Selenium-based unit tests.
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

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
