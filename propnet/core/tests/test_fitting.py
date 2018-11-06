import unittest
import os
import pandas as pd

from propnet.core.fitting import Quantity, get_sse, get_weight, \
    fit_model_scores
from propnet.core.graph import Graph
from propnet.core.materials import Material
from propnet.core.provenance import ProvenanceElement


TEST_DIR = os.path.dirname(os.path.abspath(__file__))

class FittingTests(unittest.TestCase):
    def setUp(self):
        path = os.path.join(TEST_DIR, "fitting_test_data.csv")
        test_data = pd.read_csv(path)
        graph = Graph()
        materials = [Material([Quantity("band_gap", bg)])
                     for bg in test_data['band_gap']]
        self.evaluated = [graph.evaluate(mat) for mat in materials]
        self.benchmarks = [{"refractive_index": n}
                           for n in test_data['refractive_index']]

    def test_get_sse(self):
        mats = [Material([Quantity("band_gap", n)]) for n in range(1, 5)]
        benchmarks = [{"band_gap": 1.1*n} for n in range(1, 5)]
        err = get_sse(mats, benchmarks)
        test_val = sum([0.01*n**2 for n in range(1, 5)])
        self.assertAlmostEqual(err, test_val)
        # Big dataset
        err = get_sse(self.evaluated, self.benchmarks)
        self.assertAlmostEqual(err, 173.5710251)

    def test_get_weight(self):
        q1 = Quantity("band_gap", 3.2)
        wt = get_weight(q1)
        self.assertEqual(wt, 1)
        p2 = ProvenanceElement(model="model_2", inputs=[q1])
        q2 = Quantity("refractive_index", 4, provenance=p2)
        wt2 = get_weight(q2, {"model_2": 0.5})
        self.assertEqual(wt2, 0.5)
        p3 = ProvenanceElement(model="model_3", inputs=[q2])
        q3 = Quantity("bulk_modulus", 100, provenance=p3)
        wt3 = get_weight(q3, {"model_3": 0.25, "model_2": 0.5})
        self.assertEqual(wt3, 0.125)

    def test_fit_model_scores(self):
        model_names=["band_gap_refractive_index_herve_vandamme",
                     "band_gap_refractive_index_moss",
                     "band_gap_refractive_index_ravindra",
                     "band_gap_refractive_index_reddy_ahammed",
                     "band_gap_refractive_index_reddy_anjaneyulu"]
        scores = fit_model_scores(self.evaluated, self.benchmarks,
                                  models=model_names)
        self.assertAlmostEqual(
            scores['band_gap_refractive_index_herve_vandamme'], 1.371478, 5)
