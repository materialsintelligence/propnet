import unittest
from propnet.ext.aflow import AflowAdapter
from propnet.symbols import add_builtin_symbols_to_registry
from propnet.core.registry import Registry
import os
from monty.serialization import loadfn
from pymongo.errors import ServerSelectionTimeoutError

store_file = os.environ.get('PROPNET_AFLOW_STORE_FILE')
store = None
if store_file is not None:
    store = loadfn(store_file)
    try:
        store.connect()
    except ServerSelectionTimeoutError:
        store = None


class AflowAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        Registry.clear_all_registries()
        add_builtin_symbols_to_registry()
        cls.afa_web = AflowAdapter()
        if store:
            cls.afa_store = AflowAdapter(store)
        else:
            cls.afa_store = None

    def test_get_material_for_auid(self):
        auid = 'aflow:0132ab6b9cddd429'
        materials = [self.afa_web.get_material_by_auid(auid)]
        if store:
            materials.append(self.afa_store.get_material_by_auid(auid))

        for material in materials:
            self.assertIsNotNone(material)
            self.assertEqual(list(material['external_identifier_aflow'])[0].value, auid)
            self.assertAlmostEqual(list(material['band_gap'])[0].magnitude, 0)
            self.assertIn('structure', material._quantities_by_symbol)
            self.assertIn('elastic_tensor_voigt', material._quantities_by_symbol)

    def test_get_materials_for_auids(self):
        auids = ['aflow:0132ab6b9cddd429', 'aflow:d0c93a9396dc599e']
        results = [self.afa_web.get_materials_by_auids(auids)]
        if store:
            results.append(self.afa_store.get_materials_by_auids(auids))

        for result in results:
            ybpb3, nacl = result
            self.assertEqual(list(ybpb3['external_identifier_aflow'])[0].value, auids[0])
            self.assertEqual(list(nacl['external_identifier_aflow'])[0].value, auids[1])

    def test_get_properties(self):
        auids = ['aflow:0132ab6b9cddd429', 'aflow:d0c93a9396dc599e']
        results = [[m for m in self.afa_web.get_properties_from_web(criteria={'auid': {'$in': auids}},
                                                                    properties=None)]]

        if store:
            results.append([m for m in self.afa_store.get_properties_from_store(criteria={'auid': {'$in': auids}},
                                                                                properties=None)])
        for result in results:
            # Store queries may not come back in order
            for material in result:
                if material['auid'] == auids[0]:
                    self.assertAlmostEqual(material['Egap'], 0)
                else:
                    self.assertEqual(material['compound'], 'Cl1Na1')


if __name__ == '__main__':
    unittest.main()
