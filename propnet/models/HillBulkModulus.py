from propnet.core.models import AbstractModel


class HillBulkModulus(AbstractModel):
    def plug_in(self, symbol_values):
        s = symbol_values['S']
        c = symbol_values['C']
        b1 = 1 / (s[0][0] + s[1][1] + s[2][2] + 2 * (s[0][1] + s[1][2] + s[0][2]))
        b2 = 1 / 9 * (c[0][0] + c[1][1] + c[2][2]) + 2 / 9 * (c[0][1] + c[1][2] + c[0][2])
        return {'B': 1/2 * (b1+b2)}
