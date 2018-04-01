import unittest
import os

from glob import glob
from monty.serialization import loadfn

import propnet.models as models

from propnet.models import DEFAULT_MODEL_NAMES
from propnet.core.models import *


class ModelTest(unittest.TestCase):

    def test_instantiate_all_models(self):
        models_to_test = []
        for model_name in DEFAULT_MODEL_NAMES:
            try:
                model = getattr(models, model_name)()
                models_to_test.append(model_name)
            except Exception as e:
                self.fail('Failed to load model {}: {}'.format(model_name, e))

    def test_evaluate(self):
        test_data = glob(os.path.join(os.path.dirname(__file__), '../../models/test_data/*.json'))
        for f in test_data:
            model_name = os.path.splitext(os.path.basename(f))[0]
            if model_name in DEFAULT_MODEL_NAMES:
                model = getattr(models, model_name)()
                self.assertTrue(model.test())
            else:
                raise ValueError("Model matching test data not found: {}".format(model_name))
