from propnet.core.models import AbstractAnalyticalModel
from typing import *


class RefractiveIndexfromRelPerm(AbstractAnalyticalModel):
    """ """

    @property
    def title(self):
        """ """
        return "Refractive index, relative permeability and permittivity"

    @property
    def tags(self):
        """ """
        return ["optical"]

    @property
    def description(self):
        """ """

    @property
    def references(self):
        """

        Args:

        Returns:
          @misc{refractive2017,
          title={Refractive index},
          url={https://en.wikipedia.org/wiki/Refractive_index#Relative_permittivity_and_permeability},
          year={2017},
          month={Dec}}

        """

    @property
    def symbol_mapping(self):
        """ """
        return {
            'n': 'refractive_index',
            'Ur': 'relative_permittivity',
            'Er': 'relative_permeability'
        }

    @property
    def connections(self):
        """ """
        return {
            'n': {'Ur', 'Er'}
        }

    @property
    def equations(self):
        """ """
        return ["n - sqrt(Ur*Er)"]