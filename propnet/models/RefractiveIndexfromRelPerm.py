from propnet.core.models import AbstractAnalyticalModel, validate_evaluate

class RefractiveIndexfromRelPerm(AbstractAnalyticalModel):

    @property
    def title(self):
        return "Relate Refractive Index with relative permittivity and relative permeability of a material."

    @property
    def tags(self):
        return ["stub"]

    @property
    def description(self):
        return """
        The refractive index is the geometric mean of the relative permittivity and the relative permeability.
        """

    @property
    def references(self):
        return []

    @property
    def symbol_mapping(self):
        return {
            'n': 'refractive_index',
            'Ur': 'relative_permittivity',
            'Er': 'relative_permeability'
        }

    @property
    def constraint_properties(self):
        return None

    @property
    def inputs_are_valid(self, input_props: Dict[str, Any]):
        return None

    @property
    def output_conditions(self, symbol_out: str):
        return None;

    @property
    def test_sets(self):
        return {}

    @property
    def equations(self):
        return ["n - sqrt(Ur*Er)"]
