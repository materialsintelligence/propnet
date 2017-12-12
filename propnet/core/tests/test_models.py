import unittest

import propnet.models as models

from propnet.models import all_model_names
from propnet.core.models import *


class AnalyticalModelTest(unittest.TestCase):

    def testInstantiateAllModels(self):

        models_to_test = []

        for model_name in all_model_names:
            try:
                model = getattr(models, model_name)()
                models_to_test.append(model_name)
            except Exception as e:
                raise Exception('Failed to load model {}: {}'.format(model_name, e))

        for model_name in models_to_test:
            model = getattr(models, model_name)()
            if len(model.test_sets):
                model.test()