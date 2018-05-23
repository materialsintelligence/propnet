from propnet.core.models import AbstractModel

from pymatgen.analysis.cost.cost import CostDBElements, CostAnalyzer


class Cost(AbstractModel):

    cost_analyzer = CostAnalyzer(CostDBElements())

    def plug_in(self, symbol_values):

        formula = symbol_values['formula']

        return {
            'cost_per_kg': self.cost_analyzer.get_cost_per_kg(formula),
            'cost_per_mol': self.cost_analyzer.get_cost_per_mol(formula)
        }