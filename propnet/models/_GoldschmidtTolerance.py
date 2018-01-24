from propnet.core.models import AbstractAnalyticalModel
from typing import *


class GoldschmidtTolerance(AbstractAnalyticalModel):
    """ """

    @property
    def title(self):
        """ """
        return "Find Goldchmidt Tolerance Factor from ionic radii"

    @property
    def tags(self):
        """ """
        return ["stability"]

    @property
    def description(self):
        return """The Goldchmidt Tolerance Factor indicates stability of perovskites.

        """

    @property
    def references(self):
        """ """
        return []

    @property
    def symbol_mapping(self):
        """ """
        return {
            't': 'goldschmidt_tolerance_factor',
            'r_cation_A': 'ionic_radius_a',
            'r_cation_B': 'ionic_radius_b',
            'r_anion': 'ionic_radius'
        }

    @property
    def constants(self):
        return {
            'r_anion': (100, "pm") # or whatever this is
        }

    @property
    def connections(self):
        """ """
        return {
            't': {'r_cation_A', 'r_cation_B'}
        }

    @property
    def equations(self):
        """ """
        return ["t - (r_cation_A + r_cation_B)/(2**.5 * (rB + r_anion))"]