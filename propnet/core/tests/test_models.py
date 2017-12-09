import unittest
from propnet.core.models import *
from propnet.models import all_model_names


class AnalyticalModelTest(unittest.TestCase):

    def testAbstractAnalyticalModel(self):

        IsotropicElasticModuli.