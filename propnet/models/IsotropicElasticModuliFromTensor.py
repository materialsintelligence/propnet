from propnet.core.models import AbstractModel, validate_evaluate
from pymatgen.analysis.elasticity import ElasticTensor

class IsotropicElasticModuli(AbstractModel):

    @property
    def title(self):
        return "Calculate isotropic elastic moduli from elastic tensor"

    @property
    def tags(self):
        return ["mechanical"]

    @property
    def description(self):
        return "Description here."

    @property
    def symbol_mapping(self):
        return {
            'Eij': 'elastic_tensor_voigt',
            'E': 'youngs_modulus',
            #'G': 'shear_modulus',
            #'n': 'poisson_ratio',
            #'K': 'bulk_modulus',
            #'l': 'lame_first_parameter',
            #'M': 'p_wave_modulus'
        }

    @property
    def valid_inputs(self):
        return ['Eij']

    @property
    def valid_outputs(self):
        return ['E']

    @validate_evaluate
    def evaluate(self, symbols_and_values_in, symbol_out):

        tensor = ElasticTensor.from_voigt(symbols_and_values_in.get('E_ij'))

        if symbol_out == 'E':
            return tensor.y_mod

        return None