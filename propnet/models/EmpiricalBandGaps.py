from propnet.core.models import *


class EmpiricalBandGap(AbstractAnalyticalModel):

    @property
    def title(self):
        return "Estimate an empirical band gap from a GW band gap"

    @property
    def tags(self):
        return ["electronic"]

    @property
    def description(self):
        return """
        """

    @property
    def references(self):
        return ["""DOI: 10.1021/acs.jpcc.7b07421"""]

    @property
    def symbol_mapping(self):
        return {
            'E_g': 'band_gap'
        }

    @property
    def test_sets(self):
        return [
            {'E_g_expt': 2.34}
        ]

    def equations(self):
        return [
            "((3 * K * (3 * K - E)) / (9 * K - E)) - l",
            "((3 * K * E) / (9 * K - E)) - G",
            "((3 * K - E) / (6 * K)) - n",
            "((3 * K) * (3 * K + E) / (9 * K - E)) - M"
        ]