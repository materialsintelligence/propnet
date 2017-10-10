from propnet.core.models import AbstractAnalyticalModel, validate_evaluate


class IsotropicElasticModuli(AbstractAnalyticalModel):

    @property
    def title(self):
        return "Convert between isotropic elastic moduli"

    @property
    def tags(self):
        return ["mechanical"]

    @property
    def description(self):
        return """
        The isotropic elastic moduli are ...
        """

    @property
    def references(self):
        return {}

    @property
    def symbol_mapping(self):
        return {
            'E': 'youngs_modulus',
            'G': 'shear_modulus',
            'n': 'poisson_ratio',
            'K': 'bulk_modulus',
            'l': 'lame_first_parameter',
            'M': 'p_wave_modulus'
        }

    @property
    def valid_outputs(self):
        return ['E', 'G', 'n', 'K', 'l', 'M']

    def equations(self, E, G, n, K, l, M):
        return [
            ((3 * K * (3 * K - E)) / (9 * K - E)) - l,
            ((3 * K * E) / (9 * K - E)) - G,
            ((3 * K - E) / (6 * K)) - n,
            ((3 * K) * (3 * K + E) / (9 * K - E)) - M
        ]

    @property
    def test_sets(self):
        return [
            {'K': (0, "GPa"),
             'E': (0, "GPa"),
             'G': (0, "GPa")}
        ]

    @property
    def constraints(self):
        return {
            # '*': {'positive': True}, # add a special 'all' operator?
            'E': {'positive': True, 'rational': True, 'finite': True}  # add some of these to defaults...
        }