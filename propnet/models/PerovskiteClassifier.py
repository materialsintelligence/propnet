from propnet.core.models import AbstractModel
from propnet.core.quantity import Quantity

class PerovskiteClassifier(AbstractModel):

    def plug_in(self, symbol_values):

        # placeholder, a little dumb
        # will be partly replaced with CrystalPrototypeClassifier

        structure = symbol_values['s']

        # support other anions too?
        if 'O' not in [sp.symbol for sp in structure.types_of_specie]:
            return None

        if structure.composition.anonymized_formula != 'ABC3':
            return None

        radii = []
        for sp in structure.types_of_specie:
            if sp.symbol != 'O':
                radii.append(float(sp.average_ionic_radius))

        return {
            'r_A': max(radii),
            'r_B': min(radii)
        }