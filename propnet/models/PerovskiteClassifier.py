from propnet.core.models import AbstractModel
from propnet.core.symbols import Symbol

class PerovskiteClassifier(AbstractModel):

    def _evaluate(self, symbol_values):

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
                radii.append(sp.ionic_radius)

        return {
            'r_A': Symbol('ionic_radius', max(radii), tags='anion_A'),
            'r_B': Symbol('ionic_radius', min(radii), tags='anion_B')
        }