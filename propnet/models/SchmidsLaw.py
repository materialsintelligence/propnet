from propnet.core.models import AbstractAnalyticalModel, validate_evaluate
from typing import *


class SchmidsLaw(AbstractAnalyticalModel):

    @property
    def title(self):
        return "Schmid's Law"

    @property
    def tags(self):
        return ["mechanical"]

    @property
    def description(self):
        return """Schmid's Law states that the critically resolved shear stress is equal
        to the stress applied to the material multipled by a geometric factor combining
        angles of the glide plane and glide direction.
        """

    @property
    def references(self):
        return """
        
        @misc{schmid2017,
        title={Schmid's law},
        url={https://en.wikipedia.org/wiki/Schmid%27s_law},
        journal={Wikipedia},
        publisher={Wikimedia Foundation},
        year={2017},
        month={Dec}}
        
        """

    @property
    def symbol_mapping(self):
        return {
            'T': 'shear_modulus',
            'm': 'schmid_factor',
            'σ': 'applied_stress'
        }

    @property
    def connections(self):
        return {
            'T': {'m', 'σ'}
        }

    @property
    def equations(self):
        return ["T-mσ"]
