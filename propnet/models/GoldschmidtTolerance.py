from propnet.core.models import AbstractAnalyticalModel
import sympy as sp

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
            'r0': 'radius_anion',
            'rA': 'radius_A_cation',
            'rB': 'radius_B_cation'
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
        return ["t - (rA + rB)/(2**.5 * (rB + r0))"]

    @property
    def output_conditions(self, symbol_out: str):
        return None

    @property
    def test_sets(self):
        return {}
