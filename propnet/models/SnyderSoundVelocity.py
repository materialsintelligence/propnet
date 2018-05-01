from propnet.core.models import AbstractModel
from pymatgen.analysis.elasticity import ElasticTensor

class SnyderSoundVelocity(AbstractModel):

    def _evaluate(self, symbol_values):
        return NotImplementedError