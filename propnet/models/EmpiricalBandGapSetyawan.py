from propnet.core.models import *

class EmpiricalBandGapSetyawan(AbstractAnalyticalModel):
    """ """

    @property
    def title(self):
        """ """
        return "Estimate experimental band gaps from DFT (PBE) calculations"

    @property
    def tags(self):
        """ """
        return ["electronic"]

    @property
    def revision(self):
        """ """
        return 1

    @property
    def description(self):
        return """Band gaps estimated using Density Functional Theory (and the PBE exchange-correlation
        functional) typically underestimate the true band gap. A linear least squares fit
        can provide an estimate of the true band gap from the PBE gap (R² = 0.886 when tested
        against 100 compounds with gaps between ~1 and ~12 eV).
        """

    @property
    def references(self):
        return """
          @article{Setyawan2011,
          doi = {10.1021/co200012w},
          year  = {2011},
          month = {July},
          publisher = {American Chemical Society ({ACS})},
          volume = {13},
          number = {4},
          pages = {382--390},
          author = {Wahyu Setyawan and Romain M. Gaume and Stephanie Lam and Robert S. Feigelson and Stefano Curtarolo},
          title = {High-Throughput Combinatorial Database of Electronic Band Structures for Inorganic Scintillator Materials},
          journal = {{ACS} Combinatorial Science}}

        """

    @property
    def symbol_mapping(self):
        """ """
        return {
            'E_g_expt': 'band_gap',
            'E_g_PBE': 'band_gap_pbe'
        }

    @property
    def connections(self):
        """ """
        return {
            'E_g_expt': {'E_g_PBE'}
        }

    @property
    def equations(self):
        """ """
        return [
            "E_g_expt - 1.348*E_g_pbe + 0.913"
        ]