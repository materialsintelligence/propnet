import unittest
import os

from glob import glob
from monty.serialization import loadfn

import propnet.models as models

from propnet.models import all_model_names
from propnet.core.models import *


class ModelTest(unittest.TestCase):

    def testInstantiateAllModels(self):
        models_to_test = []
        for model_name in all_model_names:
            try:
                model = getattr(models, model_name)()
                models_to_test.append(model_name)
            except Exception as e:
                raise Exception('Failed to load model {}: {}'.format(model_name, e))

    def testEvaluate(self):
        test_data = glob(os.path.join(os.path.dirname(__file__), '../../models/test_data/*.json'))
        for f in test_data:
            model_name = os.path.splitext(os.path.basename(f))[0]
            if model_name in all_model_names:
                model = getattr(models, model_name)()
                model_test_data = loadfn(f)
                for d in model_test_data:
                    model_outputs = model.evaluate(d['inputs'])
                    for k, v in d['outputs'].items():
                        self.assertAlmostEqual(model_outputs[k], v)
            else:
                raise ValueError("Model not found: {}".format(model_name))
