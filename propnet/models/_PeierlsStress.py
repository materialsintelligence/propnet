from propnet.core.models import AbstractAnalyticalModel


class PeierlsStress(AbstractAnalyticalModel):
    """ """

    @property
    def title(self):
        """ """
        return "Peierls-Nabarro Stress for Dislocation Slip"

    @property
    def tags(self):
        """ """
        return ["dislocations"]

    @property
    def description(self):
        return """Peierls stress is the force required to move a dislocation within a plane of atoms in a unit cell.

        """

    @property
    def references(self):
        return """

        Returns:
          @misc{peierls2017,
          title={Peierls stress},
          url={https://en.wikipedia.org/wiki/Peierls_stress},
          year={2017},
          month={Dec}}

        """

    @property
    def symbol_mapping(self):
        """ """
        return {
            'T_pn': 'peierls_stress',
            'G': 'shear_modulus',
            'ν': 'poisson_ratio',
            'a': 'interplanar_spacing',
            'b': 'interatomic_spacing'
        }

    @property
    def connections(self):
        """ """
        return {
            'T_pn': {'G', 'ν', 'a', 'b'}
        }

    @property
    def equations(self):
        """ """
        return ["T_pn - G*sp.E**(-2*sp.pi*a/(1-ν)/b)"]
