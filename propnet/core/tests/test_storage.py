import unittest
import os

import numpy as np

from propnet.core.storage import StorageQuantity, ProvenanceStore, ProvenanceStoreQuantity
from propnet.core.symbols import Symbol
from propnet.core.quantity import QuantityFactory
from propnet.core.provenance import ProvenanceElement
from propnet import ureg


TEST_DIR = os.path.dirname(os.path.abspath(__file__))


class StorageTest(unittest.TestCase):
    def setUp(self):
        # Most of this setup is verbatim from the GraphTest class
        symbols = StorageTest.generate_canonical_symbols()

        a = [QuantityFactory.create_quantity(symbols['A'], 19),
             QuantityFactory.create_quantity(symbols['A'], 23)]
        b = [QuantityFactory.create_quantity(symbols['B'], 38,
                                             provenance=ProvenanceElement(model='model1',
                                                                          inputs=[a[0]])),
             QuantityFactory.create_quantity(symbols['B'], 46,
                                             provenance=ProvenanceElement(model='model1',
                                                                          inputs=[a[1]]))]
        c = [QuantityFactory.create_quantity(symbols['C'], 57,
                                             provenance=ProvenanceElement(model='model1',
                                                                          inputs=[a[0]])),
             QuantityFactory.create_quantity(symbols['C'], 69,
                                             provenance=ProvenanceElement(model='model1',
                                                                          inputs=[a[1]]))]
        g = [QuantityFactory.create_quantity(symbols['G'], 95,
                                             provenance=ProvenanceElement(model='model2',
                                                                          inputs=[a[0]])),
             QuantityFactory.create_quantity(symbols['G'], 115,
                                             provenance=ProvenanceElement(model='model2',
                                                                          inputs=[a[1]]))]
        f = [QuantityFactory.create_quantity(symbols['F'], 266,
                                             provenance=ProvenanceElement(model='model3',
                                                                          inputs=[b[0]])),
             QuantityFactory.create_quantity(symbols['F'], 322,
                                             provenance=ProvenanceElement(model='model3',
                                                                          inputs=[b[1]]))]
        d = [QuantityFactory.create_quantity(symbols['D'], 23826,
                                             provenance=ProvenanceElement(model='model4',
                                                                          inputs=[b[0], c[0]])),
             QuantityFactory.create_quantity(symbols['D'], 28842,
                                             provenance=ProvenanceElement(model='model4',
                                                                          inputs=[b[0], c[1]])),
             QuantityFactory.create_quantity(symbols['D'], 28842,
                                             provenance=ProvenanceElement(model='model4',
                                                                          inputs=[b[1], c[0]])),
             QuantityFactory.create_quantity(symbols['D'], 34914,
                                             provenance=ProvenanceElement(model='model4',
                                                                          inputs=[b[1], c[1]])),
             QuantityFactory.create_quantity(symbols['D'], 70395,
                                             provenance=ProvenanceElement(model='model5',
                                                                          inputs=[c[0], g[0]])),
             QuantityFactory.create_quantity(symbols['D'], 85215,
                                             provenance=ProvenanceElement(model='model5',
                                                                          inputs=[c[0], g[1]])),
             QuantityFactory.create_quantity(symbols['D'], 85215,
                                             provenance=ProvenanceElement(model='model5',
                                                                          inputs=[c[1], g[0]])),
             QuantityFactory.create_quantity(symbols['D'], 103155,
                                             provenance=ProvenanceElement(model='model5',
                                                                          inputs=[c[1], g[1]]))]
        self.expected_quantities = a + b + c + d + f + g

    @staticmethod
    def generate_canonical_symbols():
        """
        Returns a set of Symbol objects used in testing.
        Returns: (dict<str, Symbol>)
        """
        A = Symbol('A', ['A'], ['A'], units="dimensionless", shape=[1])
        B = Symbol('B', ['B'], ['B'], units="dimensionless", shape=[1])
        C = Symbol('C', ['C'], ['C'], units="dimensionless", shape=[1])
        D = Symbol('D', ['D'], ['D'], units="dimensionless", shape=[1])
        G = Symbol('G', ['G'], ['G'], units="dimensionless", shape=[1])
        F = Symbol('F', ['F'], ['F'], units="dimensionless", shape=[1])
        return {
            'A': A,
            'B': B,
            'C': C,
            'D': D,
            'G': G,
            'F': F
        }

    def test_storage_quantity_from_quantity(self):
        quantities = self.expected_quantities
        d = quantities[10]

        storage_quantity = StorageQuantity.from_quantity(d)
        self.assertIsInstance(storage_quantity, StorageQuantity)
        # This checks value equality
        self.assertTrue(np.isclose(storage_quantity.value, d.value))
        self.assertListEqual(storage_quantity.tags, d.tags)

        self.rec_provenance_tree_check(storage_quantity.provenance, d.provenance)

    def test_from_provenance_element(self):
        quantities = self.expected_quantities
        # a = quantities[0]
        # b = quantities[2]
        # c = quantities[4]
        d = quantities[10]

        storage_provenance = ProvenanceStore.from_provenance_element(d.provenance)

        self.rec_provenance_tree_check(storage_provenance, d.provenance)

    def test_provenance_storage_quantity_from_quantity(self):
        pass

    def rec_provenance_tree_check(self, q_storage, q_original):
        self.assertIsInstance(q_storage, ProvenanceStore)
        self.assertEqual(q_storage.model, q_original.model)
        for v in q_storage.inputs or []:
            self.assertIsInstance(v, ProvenanceStoreQuantity)
            v_orig = [x for x in q_original.inputs
                      if x._internal_id == v._internal_id]
            self.assertEqual(len(v_orig), 1)
            v_orig = v_orig[0]
            self.assertTrue(np.isclose(v.value, v_orig.value))
            self.assertListEqual(v.tags, v_orig.tags)
            self.rec_provenance_tree_check(v.provenance, v_orig.provenance)
