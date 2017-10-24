from propnet.core.models import AbstractModel, validate_evaluate
from pymatgen.analysis.elasticity import ElasticTensor, ComplianceTensor

class ElasticComplianceVoigtConverter(AbstractModel):

    @property
    def title(self):
        return "Convert between elastic tensors and compliance tensors (in Voigt notation)"

    @property
    def tags(self):
        return ["stub", "mechanical"]

    @property
    def description(self):
        return """
        The compliance tensor is the inverse of elastic tensor.
        """

    @property
    def references(self):
        return []

    @property
    def symbol_mapping(self):
        return {
            'Cij': 'elastic_tensor_voigt',
            'Sij': 'compliance_tensor_voigt'
        }

    @property
    def assumption_mapping(self):
        return None

    @property
    def required_conditions(self):
        return None

    @property
    def connections(self):
        return {
            'Sij': {'Cij'},
            'Cij': {'Sij'}
        }

    @validate_evaluate
    def evaluate(self,
                 symbols_and_values_in,
                 symbol_out):

        if symbol_out == 'Cij':

            tensor = ComplianceTensor.from_voigt(symbols_and_values_in.get('Sij'))
            return tensor.elastic_tensor.voigt

        elif symbol_out == 'Sij':

            tensor = ElasticTensor.from_voigt(symbols_and_values_in.get('Cij'))
            return tensor.compliance_tensor.voigt

        return None

    @property
    def test_sets(self):
        return {}
