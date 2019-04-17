import unittest
import os

from monty.serialization import loadfn
from monty.json import jsanitize
from maggma.stores import MemoryStore
from maggma.runner import Runner

from propnet.models import add_builtin_models_to_registry
from propnet.dbtools.mp_builder import PropnetBuilder

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


class MPBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        add_builtin_models_to_registry()
        cls.materials = MemoryStore()
        cls.materials.connect()
        materials = loadfn(os.path.join(TEST_DIR, "test_materials.json"))
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


# Just here for reference, in case anyone wants to create a new set
# of test materials -jhm
def create_test_docs():
    formulas = ["BaNiO3", "Si", "Fe2O3", "Cs"]
    from maggma.advanced_stores import MongograntStore
    from monty.serialization import dumpfn
    mgstore = MongograntStore("ro:matgen2.lbl.gov/mp_prod", "materials")
    builder = PropnetBuilder(
        mgstore, MemoryStore(), criteria={"pretty_formula": {"$in": formulas},
                                          "e_above_hull": 0})
    builder.connect()
    dumpfn(list(builder.get_items()), "test_materials.json")


if __name__ == "__main__":
    unittest.main()
