from propnet.core.models import AbstractModel


class ReussBulkModulus(AbstractModel):
    def plug_in(self, symbol_values):
        s = symbol_values['S']
        return {'B': 1/(s[0][0] + s[1][1] + s[2][2] + 2*(s[0][1] + s[1][2] + s[0][2]))}
