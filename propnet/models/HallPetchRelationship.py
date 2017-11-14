from propnet.core.models import AbstractAnalyticalModel, validate_evaluate
from typing import *

class HallPetchRelationship(AbstractAnalyticalModel):

    @property
    def title(self):
        return "Hall-Petch Relationship"

    @property
    def tags(self):
        return ["stub"]

    @property
    def description(self):
        return """
        The Hall-Petch Relationship relates yield stress with average grain size.
        """

    @property
    def references(self):
        return []

    @property
    def symbol_mapping(self):
        return {
            'd': 'avg_grain_diameter',
            'k': 'strengthening_coefficient',
            'o': 'dislocation_movement_stress',
            'y': 'yield_stress'
        }

    @property
    def constraint_properties(self):
        return None

    @property
    def inputs_are_valid(self, input_props: Dict[str, Any]):
        #needs to be polycrystalline
        return True

    @property
    def output_conditions(self, symbol_out: str):
        return None

    @property
    def test_sets(self):
        return {}

    @property
    def equations(self):
        return ["y - o - k / d**(.5)"]
