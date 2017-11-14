from propnet.core.models import AbstractAnalyticalModel, validate_evaluate
import sympy as sp

class SchmidsLaw(AbstractAnalyticalModel):

    @property
    def title(self):
        return "Schmids Law"

    @property
    def tags(self):
        return ["stub"]

    @property
    def description(self):
        return """
        Schmids Law finds the resolved shear stress of an applied stress.
        """

    @property
    def references(self):
        return []

    @property
    def symbol_mapping(self):
        return {
            'T': 'resolved_shear_stress',
            'm': 'schmid_factor',
            'o': 'applied_stress'
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
        return ["t-mo"]
