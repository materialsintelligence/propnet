from propnet.core.models import AbstractAnalyticalModel

class IsotropicElasticModuli(AbstractAnalyticalModel):

    @property
    def title(self):
        return "Calculate isotropic elastic moduli from elastic tensor"

    @property
    def tags(self):
        return ["mechanical"]

    @property
    def description(self):
        return ""

    @property
    def symbol_mapping(self):
        return {
            'Eij': 'elastic_tensor_voight',
            'E': 'youngs_modulus',
            'G': 'shear_modulus',
            'n': 'poisson_ratio',
            'K': 'bulk_modulus',
            'l': 'lame_first_parameter',
            'M': 'p_wave_modulus'
        }