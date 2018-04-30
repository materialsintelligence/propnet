from propnet.core.models import AbstractModel
import numpy as np


class HillShearModulus(AbstractModel):
    def plug_in(self, symbol_values):
        s = symbol_values['S']
        c = symbol_values['C']
        g1 = 15. / (8. * s[:3, :3].trace() -
                     4. * np.triu(s[:3, :3]).sum() +
                     3. * s[3:, 3:].trace())
        g2 = 1 / 15 * (c[0][0] + c[1][1] + c[2][2]) - 1 / 15 * (c[0][1] + c[1][2] + c[2][0]) + \
                     1 / 5 * (c[3][3] + c[4][4] + c[5][5])
        return {'G': 1/2*(g1+g2)}
