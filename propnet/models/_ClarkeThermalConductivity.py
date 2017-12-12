from propnet.core.models import *


class ClarkeThermalConductivity(AbstractAnalyticalModel):
    """ """

    @property
    def title(self):
        """ """
        return "Convert between isotropic elastic moduli"

    @property
    def tags(self):
        """ """
        return ["thermal"]

    @property
    def description(self):
        """ """

    @property
    def references(self):
        """ """
                """DOI: 10.1016/j.commatsci.2015.07.029"""]

    @property
    def symbol_mapping(self):
        """ """
        return {
            'n': 'youngs_modulus',
            'E': 'shear_modulus',
            'm': 'poisson_ratio',
            'V': 'bulk_modulus',
            's': 'lame_first_parameter',
        }

    @property
    def test_sets(self):
        """ """
        return [
            {'K': (0, "GPa"),
             'E': (0, "GPa"),
             'G': (0, "GPa")}
        ]

    def equations(self):
        """ """
        return [
            "((3 * K * (3 * K - E)) / (9 * K - E)) - l",
            "((3 * K * E) / (9 * K - E)) - G",
            "((3 * K - E) / (6 * K)) - n",
            "((3 * K) * (3 * K + E) / (9 * K - E)) - M"
        ]