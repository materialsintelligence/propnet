import unittest
from propnet.ext.aflow import AflowAdapter
from propnet.symbols import add_builtin_symbols_to_registry
from propnet.core.materials import Material
from propnet.core.registry import Registry
import os
from monty.serialization import loadfn
from maggma.stores import MemoryStore
from aflow.entries import Entry
from pymatgen.core.structure import Structure

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'test_data')


class AflowAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        Registry.clear_all_registries()
        add_builtin_symbols_to_registry()
        cls.afa_web = AflowAdapter()
        store_data = loadfn(os.path.join(TEST_DATA_DIR, 'aflow_store.json'))
        store = MemoryStore()
        store.connect()
        store.update(store_data, key='auid')
        cls.afa_store = AflowAdapter(store)

    def test_get_material_for_auid(self):
        auid = 'aflow:0132ab6b9cddd429'
        materials = [self.afa_web.get_material_by_auid(auid),
                     self.afa_store.get_material_by_auid(auid)]

        for material in materials:
            self.assertIsNotNone(material)
            self.assertEqual(list(material['external_identifier_aflow'])[0].value, auid)
            self.assertAlmostEqual(list(material['band_gap'])[0].magnitude, 0)
            self.assertIn('structure', material._quantities_by_symbol)
            self.assertIn('elastic_tensor_voigt', material._quantities_by_symbol)

        materials = [self.afa_web.get_material_by_auid('garbage'),
                     self.afa_store.get_material_by_auid('garbage')]
        self.assertTrue(all(v is None for v in materials))

    def test_get_materials(self):
        auids = ['aflow:0132ab6b9cddd429', 'aflow:d0c93a9396dc599e']
        results = [self.afa_web.get_materials_by_auids(auids),
                   self.afa_store.get_materials_by_auids(auids)]

        for result in results:
            ybpb3, nacl = result
            self.assertEqual(list(ybpb3['external_identifier_aflow'])[0].value, auids[0])
            self.assertEqual(list(nacl['external_identifier_aflow'])[0].value, auids[1])

        results = [self.afa_web.get_materials_by_auids(['garbage', 'aflow:0132ab6b9cddd429']),
                   self.afa_store.get_materials_by_auids(['garbage', 'aflow:0132ab6b9cddd429'])]
        for result in results:
            self.assertIsNone(result[0])
            self.assertIsNotNone(result[1])

        results = self.afa_store.get_materials_from_store(criteria={'auid': {'$in': auids}},
                                                          properties=None)
        item = results.__next__()
        self.assertIsInstance(item, Material)
        self.assertTrue(item.symbol_quantities_dict)

        results = self.afa_web.get_materials_from_web(criteria={'auid': {'$in': auids}},
                                                      properties=None)
        item = results.__next__()
        self.assertIsInstance(item, Material)
        self.assertTrue(item.symbol_quantities_dict)

        # Ensure adapters with stores can still use web queries
        results = self.afa_store.get_materials_from_web(criteria={'auid': {'$in': auids}},
                                                        properties=None)
        item = results.__next__()
        self.assertIsInstance(item, Material)
        self.assertTrue(item.symbol_quantities_dict)

    def test_get_properties(self):
        auids = ['aflow:0132ab6b9cddd429', 'aflow:d0c93a9396dc599e']
        results = [[m for m in self.afa_web.get_properties_from_web(criteria={'auid': {'$in': auids}},
                                                                    properties=None)],
                   [m for m in self.afa_store.get_properties_from_store(criteria={'auid': {'$in': auids}},
                                                                        properties=None)]]
        for result in results:
            # Store queries may not come back in order
            for material in result:
                if material['auid'] == auids[0]:
                    self.assertAlmostEqual(material['Egap'], 0)
                else:
                    self.assertEqual(material['compound'], 'Cl1Na1')

                self.assertIn('energy_atom', material)

        results = [[m for m in self.afa_web.get_properties_from_web(criteria={'auid': {'$in': auids}},
                                                                    properties=['auid', 'Egap', 'compound'])],
                   [m for m in self.afa_store.get_properties_from_store(criteria={'auid': {'$in': auids}},
                                                                        properties=['auid', 'Egap', 'compound'])]]

        for result in results:
            # Store queries may not come back in order
            for material in result:
                if material['auid'] == auids[0]:
                    self.assertAlmostEqual(material['Egap'], 0)
                else:
                    self.assertEqual(material['compound'], 'Cl1Na1')
                self.assertNotIn('energy_atom', material)



    def test_generate_auids(self):
        auid_web_generator = self.afa_web.generate_all_auids(max_request_size=10)
        auid_web_generator_with_metadata = self.afa_web.generate_all_auids(max_request_size=10, with_metadata=True)

        num_to_check = 3
        for no_metadata, with_metadata in zip(auid_web_generator, auid_web_generator_with_metadata):
            self.assertIsInstance(no_metadata, str)
            self.assertTrue(no_metadata.startswith('aflow:'))
            self.assertIsInstance(with_metadata, dict)
            self.assertCountEqual(with_metadata.keys(), ['auid', 'aurl', 'compound', 'aflowlib_date'])
            self.assertTrue(with_metadata['auid'].startswith('aflow:'))
            num_to_check -= 1
            if num_to_check == 0:
                break

        auid_store_generator = self.afa_store.generate_all_auids_from_store()
        auid_store_generator_with_metadata = self.afa_store.generate_all_auids_from_store(with_metadata=True)

        num_to_check = 3
        for no_metadata, with_metadata in zip(auid_store_generator, auid_store_generator_with_metadata):
            self.assertIsInstance(no_metadata, str)
            self.assertTrue(no_metadata.startswith('aflow:'))
            self.assertIsInstance(with_metadata, dict)
            self.assertCountEqual(with_metadata.keys(), ['auid', 'aurl', 'compound', 'aflowlib_date'])
            self.assertTrue(with_metadata['auid'].startswith('aflow:'))
            num_to_check -= 1
            if num_to_check == 0:
                break

    def test_store_funcs_without_store(self):
        # Verify that store-based funcs throw error when no store is specified
        with self.assertRaises(ValueError):
            self.afa_web.generate_all_auids_from_store().__next__()
        with self.assertRaises(ValueError):
            self.afa_web.get_materials_from_store().__next__()
        with self.assertRaises(ValueError):
            self.afa_web.get_properties_from_store().__next__()

    def test_get_structure(self):
        auid = 'aflow:0132ab6b9cddd429'

        data = self.afa_store.store.query_one(
            criteria={'auid': auid},
            properties=['aurl', 'geometry', 'species',
                        'composition', 'positions_fractional',
                        'CONTCAR_relax_vasp'])

        structure = self.afa_store._get_structure(Entry(**data),
                                                  use_web_api=False)

        self.assertIsInstance(structure, Structure)

        del data['CONTCAR_relax_vasp']

        structure = self.afa_store._get_structure(Entry(**data),
                                                  use_web_api=False)

        self.assertIsInstance(structure, Structure)

        structure = self.afa_store._get_structure(Entry(aurl=data['aurl']),
                                                  use_web_api=True)

        self.assertIsInstance(structure, Structure)


if __name__ == '__main__':
    unittest.main()
