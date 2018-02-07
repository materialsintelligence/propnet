from propnet.core.models import AbstractModel


class GoldschmidtTolerance(AbstractModel):

    @property
    def constraints(self):
        return {
            'r_anion': lambda r_anion: 'anion' in r_anion.tags,
            'r_cation_A': lambda r_cation_A: 'cation_A' in r_cation_A.tags,
            'r_cation_B': lambda r_cation_B: 'cation_B' in r_cation_B.tags
        }
