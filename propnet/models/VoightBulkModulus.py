from propnet.core.models import AbstractModel


class VoightBulkModulus(AbstractModel):
    def plug_in(self, symbol_values):
        c = symbol_values['C']
        return {'B': 1/9*(c[0][0] + c[1][1] + c[2][2]) + 2/9*(c[0][1] + c[1][2] + c[0][2])}
