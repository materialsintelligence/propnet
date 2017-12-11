from propnet.core.models import AbstractAnalyticalModel
from typing import *


class GoldschmidtTolerance(AbstractAnalyticalModel):

    @property
    def title(self):
        return "Find Goldchmidt Tolerance Factor from ionic radii"

    @property
    def tags(self):
        return []

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
            'r_cation_A': 'ionic_radius',
            'r_cation_B': 'ionic_radius',
            's': 'crystal_class'
        }

    def constraints(self):
        return {
            'class': lambda x: x == 'perovskite',
        }
    #@property
    #def inputs_are_val#id(self, input_props: Dict[str, Any]):
    #    #needs to check if material is perovskite
    #    return True

    @property
    def equations(self):
        return ["t - (r_cation_A + r_cation_B)/(2**.5 * (rB + r_anion))"]

    @property
    def test_sets(self):
        return {}
