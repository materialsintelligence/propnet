from propnet.core.models import *


class EmpiricalBandGapsGarcia(AbstractAnalyticalModel):
    """ """

    @property
    def title(self):
        """ """
        return "Estimate experimental band gaps from GW and DFT (PBE) calculations"

    @property
    def tags(self):
        """ """
        return ["electronic"]

    @property
    def description(self):
        return """Band gaps estimated using Density Functional Theory (and the PBE exchange-correlation
        functional) typically underestimate the true band gap.
        """

    @property
    def references(self):
        return """
          @article{Garcia2017,
          doi = {10.1021/acs.jpcc.7b07421},
          year = 2017,
          month = {Aug},
          publisher = {American Chemical Society ({ACS})},
          volume = {121},
          number = {34},
          pages = {18862--18866},
          author = {Ángel Morales-García and Rosendo Valero and Francesc Illas},
          title = {An Empirical, yet Practical Way To Predict the Band Gap in Solids by Using Density Functional Band Structure Calculations},
          journal = {The Journal of Physical Chemistry C}}

        """

    @property
    def symbol_mapping(self):
        """ """
        return {
            'E_g_expt': 'band_gap',
            'E_g_PBE': 'band_gap_pbe',
            'E_g_GW': 'band_gap_gw'
        }

    @property
    def connections(self):
        """ """
        return {
            'E_g_expt': {'E_g_GW'},
            'E_g_GW': {'E_g_PBE'}
        }

    @property
    def equations(self):
        """ """
        return [
            "E_g_expt - 0.998*E_g_GW + 0.014",
            "E_g_expt - 1.358*E_g_PBE + 0.904"
        ]