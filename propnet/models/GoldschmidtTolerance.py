from propnet.core.models import AbstractAnalyticalModel
from typing import *


class GoldschmidtTolerance(AbstractAnalyticalModel):

    @property
    def title(self):
        return "Find Goldchmidt Tolerance Factor from ionic radii"

    @property
    def tags(self):
        return ["rules of thumb"]

    @property
    def description(self):
        return """
        The Goldchmidt Tolerance Factor indicates stability of perovskites.
        """

    @property
    def references(self):
        return []

    @property
    def symbol_mapping(self):
        return {
            't': 'goldchmidt_tolerance_factor',
            'r_anion': 'ionic_radius',
            'r_cation_A': 'ionic_radius_a',
            'r_cation_B': 'ionic_radius_b',
            #'structure': 'oxi_structure',
            #'crystal_prototype': 'crystal_prototype'
        }

    @property
    def constraints(self):
        return {
            'crystal_prototype': 'perovskite',
        }

    @property
    def connections(self):
        return {
            't': {'r_anion', 'r_cation_A', 'r_cation_B'}
            # split into perovskite classifier ?
        }

    @property
    def equations(self):
        return ["t - (r_cation_A + r_cation_B)/(2**.5 * (rB + r_anion))"]

    def evaluate(self):

        # extract r_cation_A
        # and r_cation_B

        return
