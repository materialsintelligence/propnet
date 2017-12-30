from propnet.core.models import AbstractModel

from pymatgen.analysis.elasticity import ElasticTensor

class ClarkeThermalConductivity(AbstractModel):
    """ """

    @property
    def title(self):
        """ """
        return "Calculate Clarke thermal conductivity"

    @property
    def tags(self):
        """ """
        return ["thermal"]

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
            't': 'thermal_conductivity'
        }


    @property
    def connections(self):
        """ """
        return {
            't': {'C_ij', 'structure'}
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