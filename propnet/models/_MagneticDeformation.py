from propnet.core.models import AbstractModel


class MagneticDeformation(AbstractModel):
    """ """

    @property
    def title(self):
        """ """
        return "Magnetic Deformation"

    @property
    def tags(self):
        """ """
        return ["magnetism"]

    @property
    def description(self):
        return """This model calculates the magnetic deformation between an idealized 'non-magnetic'
        crystal lattice and a ferromagnetic crystal lattice.
        """

    @property
    def references(self):
        """ """
        return ""

    @property
    def symbol_mapping(self):
        """ """
        return {
            'l_mag': 'lattice_unit_cell',
            'l_nonmag': 'lattice_unit_cell',
            'sigma': 'magnetic_deformation'
        }


    @property
    def connections(self):
        """ """
        return {
            'sigma': {'l_mag', 'l_nonmag'}
        }

    @property
    def constraints(self):
        return {
            'mag_is_magnetic': False,
            'nonmag_is_magnetic': True,
            ('l_mag', 'l_nonmag'): lambda l_mag, l_nonmag: l_mag.material.composition == l_nonmag.material.composition
                                                           and l_mag.material.is_magnetic == True
                                                           and l_nonmag.material.is_magnetic == False
        }

    def evaluate(self,
                 symbols_and_values_in):
        return NotImplementedError