from propnet.core.models import AbstractModel

from pymatgen.transformations.standard_transformations import AutoOxiStateDecorationTransformation

class TransformationOxiStructure(AbstractModel):
    """ """

    @property
    def title(self):
        """ """
        return "Decorate crystal structure with oxidation state"

    @property
    def tags(self):
        """ """
        return ["transformations"]

    @property
    def description(self):
        """

        Args:

        Returns:
          site using the materials analysis code pymatgen.

        """

    @property
    def references(self):
        """ """
        return ""

    @property
    def symbol_mapping(self):
        """ """
        return {
            's': 'structure',
            's_oxi': 'structure_oxi'
        }


    @property
    def connections(self):
        """ """
        return {
            's_oxi': {'s'}
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

        s = symbols_and_values_in['s']

        trans = AutoOxiStateDecorationTransformation()
        s_oxi = trans.apply_transformation(s)

        return {
            's_oxi': s_oxi
        }