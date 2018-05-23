from propnet.core.models import AbstractModel

from pymatgen.analysis.find_dimension import find_dimension
from pymatgen.analysis.structure_analyzer import get_dimensionality


class Dimensionality(AbstractModel):

    def plug_in(self, symbol_values):

        structure = symbol_values['structure']

        return {
            'dimensionality_cheon': find_dimension(structure),
            'dimensionality_gorai': get_dimensionality(structure)
        }