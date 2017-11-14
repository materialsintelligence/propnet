from propnet.core.models import AbstractAnalyticalModel, validate_evaluate
import sympy as sp

class PeierlsStress(AbstractAnalyticalModel):

    @property
    def title(self):
        return "Find Peierls Stress for a given plane."

    @property
    def tags(self):
        return ["stub"]

    @property
    def description(self):
        return """
        Peierls Stress is the force needed to move a dislocation within a plane of atoms in a unit cell.
        """

    @property
    def references(self):
        return []

    @property
    def symbol_mapping(self):
        return {
            'Tpn': 'peierls_stress',
            'G': 'shear_modulus',
            'v': 'poisson_ratio',
            'a': 'interplanar_spacing',
            'b': 'interatomic_spacing'
        }

    @property
    def constraint_properties(self):
        return None

    @property
    def inputs_are_valid(self, input_props: Dict[str, Any]):
        return True

    @property
    def output_conditions(self, symbol_out: str):
        return None

    @property
    def test_sets(self):
        return {}

    @property
    def equations(self):
        return ["Tpn - G*sp.E**(-2*sp.pi*a/(1-v)/b)"]
