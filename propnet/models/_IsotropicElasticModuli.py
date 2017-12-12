from propnet.core.models import *


class IsotropicElasticModuli(AbstractAnalyticalModel):
    """ """

    @property
    def title(self):
        """ """
        return "Convert between isotropic elastic moduli"

    @property
    def tags(self):
        """ """
        return ["mechanical"]

    @property
    def description(self):
        """

        Args:

        Returns:
          This supports conversion between sets of isotropic elastic moduli.

        """

    @property
    def references(self):
        """ """
                """URL: https://en.wikipedia.org/wiki/Young%27s_modulus"""]

    @property
    def symbol_mapping(self):
        """ """
        return {
            'E': 'youngs_modulus',
            'G': 'shear_modulus',
            'n': 'poisson_ratio',
            'K': 'bulk_modulus',
            'l': 'lame_first_parameter',
            'M': 'p_wave_modulus'
        }

    @property
    def test_sets(self):
        """ """
        return []

    def equations(self):
        """ """
        return [
            "((3 * K * (3 * K - E)) / (9 * K - E)) - l",
            "((3 * K * E) / (9 * K - E)) - G",
            "((3 * K - E) / (6 * K)) - n",
            "((3 * K) * (3 * K + E) / (9 * K - E)) - M"
        ]