from propnet.core.models import AbstractModel

from pymatgen.transformations.standard_transformations import AutoOxiStateDecorationTransformation

class IsMetallic(AbstractModel):
    """ """

    @property
    def title(self):
        """ """
        return "Determine if structure is metallic"

    @property
    def tags(self):
        """ """
        return ["stub"]

    @property
    def description(self):
        return """This model returns true if band gap is zero."""

    @property
    def references(self):
        """ """
        return ""

    @property
    def symbol_mapping(self):
        """ """
        return {
            'E_g': 'band_gap_pbe',
            'is_metallic': 'is_metallic'
        }


    @property
    def connections(self):
        """ """
        return {
            'is_metallic': {'E_g'}
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

        E_g = symbols_and_values_in['E_g']

        if E_g > 0:
            is_metallic = False
        else:
            is_metallic = True

        return {
            'is_metallic': is_metallic
        }