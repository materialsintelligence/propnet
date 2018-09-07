import unittest

from pymatgen.util.testing import PymatgenTest
from monty.serialization import loadfn
from maggma.stores import MemoryStore
from maggma.runner import Runner

from propnet.core.builder import PropnetBuilder
from propnet.ext.matproj import MPRester


class BuilderTest(unittest.TestCase):
    def setUp(self):
        self.materials = MemoryStore()
        self.materials.connect()
        formulas = ["Si", "BaNiO3", "Sn"]
        docs = []
        for n, formula in enumerate(formulas):
            structure = PymatgenTest.get_structure(formula)
            docs.append({"structure": structure.as_dict(),
                         "task_id": "mp-{}".format(n),
                         "pretty_formula": formula})
        self.materials.update(docs)
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
        mpr = MPRester()
        properties = list(mpr.mapping.keys()) + ['task_id', "pretty_formula"]
        item = mpr.query({"task_id": "mp-81"}, properties=properties)[0]
        builder = PropnetBuilder(self.materials, self.propstore)
        processed = builder.process_item(item)
        self.assertIsNotNone(processed)

    # def test_runner_pipeline(self):
    #     runner = loadfn("../../../runner.json")
    #     runner.run()
