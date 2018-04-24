from propnet.core.models import AbstractModel


class IsMetallic(AbstractModel):

    def plug_in(self, symbol_values):
        return {'is_metallic': 1 if symbol_values['E_g'] <= 0 else 0}
