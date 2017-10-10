from propnet.core.models import AbstractModel, validate_evaluate
from pymatgen.analysis.elasticity import ElasticTensor, ComplianceTensor

class ElasticComplianceConverter(AbstractModel):

    @property
    def title(self):
        return "Converter"

    @property
    def tags(self):
        return ["stub", "mechanical"]

    @property
    def description(self):
        return """
        """

    @property
    def symbol_mapping(self):
        return {
            'Eij': 'elastic_tensor_voigt',
            'Sij': 'compliance_tensor_voigt'
        }

    @property
    def connections(self):
        return [
            {'in': 'Eij', 'out': 'Sij'},
            {'in': 'Sij', 'out': 'Eij'}
        ]

    @validate_evaluate
    def evaluate(self,
                 symbols_and_values_in,
                 symbol_out):

        if symbol_out == 'Eij':

            tensor = ComplianceTensor.from_voigt(symbols_and_values_in.get('Sij'))
            return tensor.elastic_tensor.voigt

        elif symbol_out == 'Sij':

            tensor = ElasticTensor.from_voigt(symbols_and_values_in.get('Sij'))
            return tensor.compliance_tensor.voigt

        return None


