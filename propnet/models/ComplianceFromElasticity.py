from propnet.core.models import AbstractModel
import numpy as np


class ComplianceFromElasticity(AbstractModel):
    def plug_in(self, symbol_values):
        if 'C' in symbol_values.keys():
            c = symbol_values['C']
            return {'S': np.linalg.inv(c)}
        elif 'S' in symbol_values.keys():
            s = symbol_values['S']
            return {'C': np.linalg.inv(s)}
