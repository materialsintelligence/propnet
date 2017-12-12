from propnet.core.models import AbstractModel


class PerovskiteClassifier(AbstractModel):

    @property
    def title(self):
        return "Perovskite Classifier"

    @property
    def tags(self):
        return ["classifier"]

    @property
    def description(self):
        return """This model classifies whether a crystal is a perovskite, and returns information
        on the A-site ionic radius and B-site ionic radius.
        """

    @property
    def references(self):
        return ""

    @property
    def symbol_mapping(self):
        return {
            's': 'structure_oxi',
            'r_A': 'ionic_radius_a',
            'r_B': 'ionic_radius_b',
        }


    @property
    def connections(self):
        return {
            frozenset({'r_A', 'r_B'}): {'s'}
        }

    def evaluate(self,
                 symbols_and_values_in,
                 symbol_out):

        structure = symbols_and_values_in['s']

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
            'r_A': max(radii),
            'r_B': min(radii)
        }