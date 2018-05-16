import unittest
import os

from glob import glob
from monty.serialization import loadfn

import propnet.models as models

from propnet.models import DEFAULT_MODELS, DEFAULT_MODEL_NAMES
from propnet.symbols import DEFAULT_SYMBOL_TYPE_NAMES
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

    def test_model_formatting(self):

        # TODO: clean up tests (self.assertNotNone), test reference format too

        for m in DEFAULT_MODELS.values():
            m = m()
            self.assertTrue(m.name is not None and m.name.isidentifier())
            self.assertTrue(m.title is not None and m.title != 'undefined')
            self.assertTrue(m.tags is not None)
            self.assertTrue(m.description is not None)
            self.assertTrue(m.symbol_mapping is not None and isinstance(m.symbol_mapping, dict) and
                            len(m.symbol_mapping.keys()) > 0)
            for key in m.symbol_mapping.keys():
                self.assertTrue(isinstance(key, str) and key.isidentifier())
                self.assertTrue(isinstance(m.symbol_mapping[key], str) and
                                m.symbol_mapping[key] in DEFAULT_SYMBOL_TYPE_NAMES)
            self.assertTrue(m.connections is not None and isinstance(m.connections, list)
                            and len(m.connections) > 0)
            for item in m.connections:
                self.assertTrue(item is not None)
                self.assertTrue(isinstance(item, dict) and 'inputs' in item.keys() and 'outputs' in item.keys())
                self.assertTrue(item['inputs'] is not None and item['outputs'] is not None)
                self.assertTrue(isinstance(item['inputs'], list) and isinstance(item['outputs'], list))
                self.assertTrue(len(item['inputs']) > 0 and len(item['outputs']) > 0)
                for in_symb in item['inputs']:
                    self.assertTrue(in_symb is not None and isinstance(in_symb, str))
                    self.assertTrue(in_symb in m.symbol_mapping.keys())
                for out_symb in item['outputs']:
                    self.assertTrue(out_symb is not None and isinstance(out_symb, str))
                    self.assertTrue(out_symb in m.symbol_mapping.keys())

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
        L = Symbol('l', ['L'], ['L'], units=[1.0, [['centimeter', 1.0]]], shape=[1])
        A = Symbol('a', ['A'], ['A'], units=[1.0, [['centimeter', 2.0]]], shape=[1])

        class GetArea(AbstractModel):
            def __init__(self):
                AbstractModel.__init__(
                    self,
                    metadata={
                        'symbol_mapping': {'l1': 'l', 'l2': 'l', 'a': 'a'},
                        'connections': [{'inputs': ['l1', 'l2'], 'outputs': ['a']}],
                        'equations': ['a - l1 * l2']
                    },
                    symbol_types={'l': L, 'a': A}
                )

        model = GetArea()
        out = model.evaluate({'l1': 1 * ureg.Quantity.from_tuple([1.0, [['meter', 1.0]]]),
                              'l2': 2 * L.units})

        self.assertTrue(math.isclose(out['a'].magnitude, 200.0))
        self.assertTrue(out['a'].units == A.units)

    def test_example_code_helper(self):

        example_model = models.SemiEmpiricalMobility()

        example_code = """from propnet.models import SemiEmpiricalMobility

m_e = 0.009  # electron_effective_mass in 1.0 dimensionless
K = 64  # bulk_modulus in 1.0 gigapascal

model = SemiEmpiricalMobility()
model.evaluate({
\t'm_e': m_e
\t'K': K
})  # returns {'mu_e': 8994.92312225673}
"""

        self.assertEqual(example_model._example_code, example_code)
