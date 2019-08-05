import unittest
import os

from monty.serialization import loadfn
from monty.json import jsanitize
from maggma.stores import MemoryStore
from maggma.runner import Runner

from propnet.models import add_builtin_models_to_registry
from propnet.dbtools.mp_builder import PropnetBuilder

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'test_data')


class MPBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        add_builtin_models_to_registry()
        cls.materials = MemoryStore()
        cls.materials.connect()
        materials = loadfn(os.path.join(TEST_DATA_DIR, "test_materials.json"))
        materials = jsanitize(materials, strict=True, allow_bson=True)
        cls.materials.update(materials)
        cls.propstore = None

    def setUp(self):
        self.propstore = MemoryStore()
        self.propstore.connect()

    def test_serial_runner(self):
        builder = PropnetBuilder(self.materials, self.propstore)
        runner = Runner([builder])
        runner.run()

    def test_multiproc_runner(self):
        builder = PropnetBuilder(self.materials, self.propstore)
        runner = Runner([builder], max_workers=2)
        runner.run()

    def test_process_item(self):
        item = self.materials.query_one(criteria={"pretty_formula": "Cs"})
        builder = PropnetBuilder(self.materials, self.propstore)
        processed = builder.process(item)
        self.assertIsNotNone(processed)
        # Ensure vickers hardness gets populated
        self.assertIn("vickers_hardness", processed)
        if 'created_at' in item.keys():
            date_value = item['created_at']
        else:
            date_value = ""

        # Check that provenance values propagate correctly
        current_quantity = processed['vickers_hardness']['quantities'][0]
        at_deepest_level = False
        while not at_deepest_level:
            current_provenance = current_quantity['provenance']
            if current_provenance['inputs'] is not None:
                self.assertEqual(current_provenance['source']['source'],
                                 "propnet")
                self.assertEqual(current_provenance['source']['source_key'],
                                 current_quantity['internal_id'])
                self.assertNotIn(current_provenance['source']['date_created'],
                                 ("", None))
                current_quantity = current_provenance['inputs'][0]
            else:
                self.assertEqual(current_provenance['source']['source'],
                                 "Materials Project")
                self.assertEqual(current_provenance['source']['source_key'],
                                 item['task_id'])
                self.assertEqual(current_provenance['source']['date_created'],
                                 date_value)
                at_deepest_level = True

    def test_deprecated_filter(self):
        # Check default (False)
        builder = PropnetBuilder(self.materials, self.propstore)
        self.assertFalse(builder.include_deprecated)

        # Check False explicitly
        builder = PropnetBuilder(self.materials, self.propstore, include_deprecated=False)
        builder.connect()
        is_deprecated = [item['deprecated'] for item in builder.get_items()]
        self.assertTrue(all(not v for v in is_deprecated))

        # Check True explicitly
        builder = PropnetBuilder(self.materials, self.propstore, include_deprecated=True)
        builder.connect()
        is_deprecated = [item['deprecated'] for item in builder.get_items()]
        self.assertTrue(any(is_deprecated))

    def test_sandboxed_filter(self):
        # Check default (False)
        builder = PropnetBuilder(self.materials, self.propstore)
        self.assertFalse(builder.include_sandboxed)

        # Check False explicitly
        builder = PropnetBuilder(self.materials, self.propstore, include_sandboxed=False)
        builder.connect()
        is_in_core = ['core' in item['sbxn'] for item in builder.get_items()]
        self.assertTrue(all(v for v in is_in_core))

        # Check True explicitly
        builder = PropnetBuilder(self.materials, self.propstore, include_sandboxed=True)
        builder.connect()
        is_not_in_core = ['core' not in item['sbxn'] for item in builder.get_items()]
        self.assertTrue(any(is_not_in_core))


if __name__ == "__main__":
    unittest.main()
