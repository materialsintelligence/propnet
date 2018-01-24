from propnet.core.models import AbstractModel

from pymatgen.analysis.elasticity.elastic import ElasticTensor

class ClarkeThermalConductivity(AbstractModel):

   def evaluate(self, symbols):

       tensor = ElasticTensor.from_voigt(symbols["C_ij"])
       structure = symbols["_structure"]

       return {'t': tensor.clarke_thermalcond(structure)}