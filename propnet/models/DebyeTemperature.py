from propnet.core.models import AbstractModel

from pymatgen.analysis.elasticity import ElasticTensor

class DebyeTemperature(AbstractModel):
    """ """

    @property
    def title(self):
        """ """
        return "Calculate Debye temperature"

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
            'd': 'debye_temperature'
        }


    @property
    def connections(self):
        """ """
        return {
            'd': {'C_ij', 'structure'}
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

        structure = symbols_and_values_in['structure']
        cij = symbols_and_values_in['C_ij']

        elastic_tensor = ElasticTensor.from_voigt(cij)

        debye_temp = elastic_tensor.debye_temperature(structure)

        return {
            'd': debye_temp
        }