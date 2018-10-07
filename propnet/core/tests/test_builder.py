import unittest

from monty.serialization import loadfn
from monty.json import jsanitize
from maggma.stores import MemoryStore, MongoStore
from maggma.runner import Runner

from propnet.core.builder import PropnetBuilder


class BuilderTest(unittest.TestCase):
    def setUp(self):
        self.materials = MemoryStore()
        self.materials.connect()
        materials = loadfn("test_materials.json")
        materials = jsanitize(materials, strict=True, allow_bson=True)
        self.materials.update(materials)
        self.propstore = MemoryStore()
        self.propstore.connect()

    def test_serial_runner(self):
        builder = PropnetBuilder(self.materials, self.propstore)
        runner = Runner([builder])
        runner.run()

    def test_multiproc_runner(self):
        builder = PropnetBuilder(self.materials, self.propstore)
        runner = Runner([builder])
        runner.run()

    def test_process_item(self):
        item = self.materials.query_one(criteria={"pretty_formula": "Cs"})
        builder = PropnetBuilder(self.materials, self.propstore)
        processed = builder.process_item(item)
        self.assertIsNotNone(processed)
        # Ensure vickers hardness gets populated
        self.assertIn("vickers_hardness", processed)

    # @unittest.skipIf(not os.path.isfile("runner.json"), "No runner file")
    # def test_runner_pipeline(self):
    #     from monty.serialization import loadfn
    #     runner = loadfn("runner.json")
    #     runner.builders[0].connect()
    #     items = list(runner.builders[0].get_items())
    #     processed = runner.builders[0].process_item(items[0])
    #     runner.run()

    # Just here for reference, in case anyone wants to create a new set
    # of test materials -jhm
    @unittest.skipIf(True, "Skipping test materials creation")
    def create_test_docs(self):
        formulas = ["BaNiO3", "Si", "Fe2O3", "Cs"]
        from maggma.advanced_stores import MongograntStore
        from monty.serialization import dumpfn
        mgstore = MongograntStore("ro:matgen2.lbl.gov/mp_prod", "materials")
        builder = PropnetBuilder(
            mgstore, self.propstore, criteria={"pretty_formula": {"$in": formulas},
                                               "e_above_hull": 0})
        builder.connect()
        items = list(builder.get_items())

        dumpfn(list(builder.get_items()), "test_materials.json")
