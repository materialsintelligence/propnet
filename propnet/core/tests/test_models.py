import unittest
import os

from glob import glob
from monty.serialization import loadfn

import propnet.models as models

from propnet.models import DEFAULT_MODEL_NAMES
from propnet.core.models import *
from propnet.core.symbols import *


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

    def test_unit_handling(self):
        """
        Tests unit handling with a simple model that calculates the area of a rectangle as the
        product of two lengths.

        In this case the input lengths are provided in centimeters and meters.
        Tests whether the input units are properly coerced into canonical types.
        Tests whether the output units are properly set.
        Tests whether the model returns as predicted.
        Returns:
            None
        """
        L = SymbolType('L', [1.0, [['centimeter', 1.0]]], ['L'], ['L'], [1], '', validate=False)
        A = SymbolType('A', [1.0, [['centimeter', 2.0]]], ['A'], ['A'], [1], '', validate=False)

        class GetArea(AbstractModel):
            def __init__(self):
                AbstractModel.__init__(
                    self,
                    metadata={
                        'symbol_mapping': {'l1': 'L', 'l2': 'L', 'a': 'A'},
                        'connections': [{'inputs': ['l1', 'l2'], 'outputs': ['a']}],
                        'equations': ['a - l1 * l2']
                    },
                    symbol_types={
                        'L': L, 'A': A
                    }
                )

        model = GetArea()
        out = model.evaluate({'l1': 1 * ureg.Quantity.from_tuple([1.0, [['meter', 1.0]]]), 'l2': 2 * L.units})

        self.assertTrue(math.isclose(out['a'].magnitude, 200.0))
        self.assertTrue(out['a'].units == A.units)
