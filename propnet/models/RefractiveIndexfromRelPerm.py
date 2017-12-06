from propnet.core.models import AbstractAnalyticalModel
from typing import *


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
    def equations(self):
        return ["n - sqrt(Ur*Er)"]

    @property
    def test_sets(self):
        return {}