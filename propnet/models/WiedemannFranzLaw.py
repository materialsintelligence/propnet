from propnet.core.models import AbstractAnalyticalModel
from typing import *

class WiedemannFranzLaw(AbstractAnalyticalModel):

    @property
    def title(self):
        return "Wiedemann-Franz Law"

    @property
    def tags(self):
        return ["stub"]

    @property
    def description(self):
        return """
        The Wiedemann-Franz Law states that the ratio of the
        electronic component of thermal conductivity to electrical
        conductivity is proportional to temperature (for a metal).
        """

    @property
    def references(self):
        return []

    @property
    def symbol_mapping(self):
        return {
            'k': 'electronic_thermal_conductivity',
            'T': 'temperature',
            'o': 'electrical_conductivity'
        }

    @property
    def constraint_properties(self):
        return None

    @property
    def inputs_are_valid(self, input_props: Dict[str, Any]):
        #input material must be a metal
        return True

    @property
    def output_conditions(self, symbol_out: str):
        return None

    @property
    def test_sets(self):
        return {}

    @property
    def equations(self):
        return ["T * 2.44*10**-8 * o - k"]
