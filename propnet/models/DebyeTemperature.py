from propnet.core.models import AbstractModel

from pymatgen.analysis.elasticity import ElasticTensor

class DebyeTemperature(AbstractModel):

    def _evaluate(self, symbol_values):

        structure = symbol_values['structure']
        cij = symbol_values['C_ij']

        elastic_tensor = ElasticTensor.from_voigt(cij)

        debye_temp = elastic_tensor.debye_temperature(structure)

        return {
            'd': debye_temp
        }