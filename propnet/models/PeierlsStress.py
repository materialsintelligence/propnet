from propnet.core.models import AbstractAnalyticalModel
from typing import *


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
    def test_sets(self):
        return []

    @property
    def equations(self):
        return ["Tpn - G*sp.E**(-2*sp.pi*a/(1-v)/b)"]
