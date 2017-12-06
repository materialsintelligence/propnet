from propnet.core.models import AbstractAnalyticalModel
from typing import *


class GoldchmidtTolerance(AbstractAnalyticalModel):

    @property
    def title(self):
        return "Find Goldchmidt Tolerance Factor from ionic radii"

    @property
    def tags(self):
        return ["stub"]

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
            's': 'structure'
        }

    @property
    def constraint_properties(self):
        return None

    @property
    def inputs_are_valid(self, input_props: Dict[str, Any]):
        #needs to check if material is perovskite
        return True

    @property
    def equations(self):
        return ["t - (r_cation_A + r_cation_B)/(2**.5 * (rB + r_anion))"]

    @property
    def output_conditions(self, symbol_out: str):
        return None

    @property
    def test_sets(self):
        return {}
