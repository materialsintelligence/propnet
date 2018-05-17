from propnet.core.models import AbstractModel


class VoightShearModulus(AbstractModel):
    def plug_in(self, symbol_values):
        c = symbol_values['C']
        return {'G': 1 / 15 * (c[0][0] + c[1][1] + c[2][2]) - 1 / 15 * (c[0][1] + c[1][2] + c[2][0]) +
                     1 / 5 * (c[3][3] + c[4][4] + c[5][5])}
