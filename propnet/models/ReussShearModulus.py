from propnet.core.models import AbstractModel
import numpy as np


class ReussShearModulus(AbstractModel):
    def plug_in(self, symbol_values):
        s = symbol_values['S']
        return {'G': 15. / (8. * s[:3, :3].trace() -
                     4. * np.triu(s[:3, :3]).sum() +
                     3. * s[3:, 3:].trace())}
