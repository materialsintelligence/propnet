from propnet.core.models import AbstractModel


class YoungsModulus(AbstractModel):

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
        return [{'K': 0, 'E': 0, 'G': 0}]

    @property
    def references(self):
        return []