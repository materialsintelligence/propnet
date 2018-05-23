from propnet.core.models import AbstractModel

from pymatgen.analysis.hhi.hhi import HHIModel

class HHI(AbstractModel):

    hhi_model = HHIModel()

    def plug_in(self, symbol_values):

        formula = symbol_values['formula']

        hhi_reserve, hhi_production = self.hhi_model.get_hhi(formula)

        return {
            'hhi_reserve': hhi_reserve,
            'hhi_production': hhi_production
        }