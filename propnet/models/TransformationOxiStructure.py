from propnet.core.models import AbstractModel

from pymatgen.transformations.standard_transformations import AutoOxiStateDecorationTransformation

class TransformationOxiStructure(AbstractModel):

    def _evaluate(self, symbol_values):

        s = symbol_values['s']

        trans = AutoOxiStateDecorationTransformation()
        s_oxi = trans.apply_transformation(s)

        return {
            's_oxi': s_oxi
        }