from propnet.core.models import AbstractModel

from pymatgen.analysis.elasticity import ElasticTensor

class SnyderSoundVelocity(AbstractModel):
    """ """

    @property
    def title(self):
        """ """
        return "Calculate Snyder sound velocities"

    @property
    def tags(self):
        """ """
        return ["elastic"]

    @property
    def description(self):
        return """This model currently uses the implementation in pymatgen."""

    @property
    def references(self):
        """ """
        return ""

    @property
    def symbol_mapping(self):
        """ """
        return {
            'C_ij': 'elastic_tensor_voigt',
            'structure': 'structure',
            'v_ac': 'snyder_acoustic_sound_velocity',
            'v_op': 'snyder_optical_sound_velocity',
            'v_tot': 'snyder_total_sound_velocity'
        }


    @property
    def connections(self):
        """ """
        return {
            'v_ac': {'C_ij', 'structure'},
            'v_op': {'C_ij', 'structure'},
            #'v_tot': {'v_op', 'v_ac'} # TODO: think about this situation
        }

    def evaluate(self,
                 symbols_and_values_in,
                 symbol_out):
        """

        Args:
          symbols_and_values_in: 
          symbol_out: 

        Returns:

        """

        return