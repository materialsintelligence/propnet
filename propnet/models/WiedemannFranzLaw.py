from propnet.core.models import AbstractAnalyticalModel
from typing import *

class WiedemannFranzLaw(AbstractAnalyticalModel):
    """ """

    @property
    def title(self):
        """ """
        return "Wiedemann-Franz Law"

    @property
    def tags(self):
        """ """
        return ["thermal"]

    @property
    def description(self):
        return """The Wiedemann-Franz Law states that the ratio of the
        electronic component of thermal conductivity to electrical
        conductivity is proportional to temperature (for a metal).
        """

    @property
    def references(self):
        return """
          @misc{widemann2017,
          title={Thermal Conductivity and the Wiedemann-Franz Law},
          url={http://hyperphysics.phy-astr.gsu.edu/hbase/thermo/thercond.html#c2},
          year={2017},
          month={Dec}}

        """

    @property
    def symbol_mapping(self):
        """ """
        return {
            'k': 'electronic_thermal_conductivity',
            'T': 'temperature',
            'o': 'electrical_conductivity',
            'is_metallic': 'is_metallic'
        }

    @property
    def constraints(self):
        """ """
        return {
            'is_metallic': lambda is_metallic: is_metallic is True,
        }

    @property
    def connections(self):
        """ """
        return {
            'k': {'T', 'o', 'is_metallic'}
        }

    @property
    def equations(self):
        """ """
        return ["T * 2.44*10**-8 * o - k"]
