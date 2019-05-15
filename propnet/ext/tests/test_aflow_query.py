import unittest
from propnet.ext.aflow import AflowAPIQuery


class AflowAPITest(unittest.TestCase):
    def test_reduce_batch(self):
        auids = ['aflow:0132ab6b9cddd429', 'aflow:d0c93a9396dc599e']
        query = AflowAPIQuery.from_pymongo(criteria={'auid': {'$in': auids}},
                                           properties=['Egap', 'energy_atom'],
                                           request_size=2)

        data = query._request_with_smaller_batch(1, 2)

        self.assertEqual(len(data), len(auids))

    def test_reduce_properties(self):
        auids = ['aflow:0132ab6b9cddd429', 'aflow:d0c93a9396dc599e']
        query = AflowAPIQuery.from_pymongo(criteria={'auid': {'$in': auids}},
                                           properties=['Egap', 'energy_atom'],
                                           request_size=2)

        data = query._request_with_fewer_props(1, 2)
        self.assertEqual(len(data), len(auids))
        for material in data.values():
            self.assertIn('Egap', material)
            self.assertIn('energy_atom', material)
