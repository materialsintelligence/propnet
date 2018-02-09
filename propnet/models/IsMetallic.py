from propnet.core.models import AbstractModel


class IsMetallic(AbstractModel):

    def _evaluate(self, symbol_values):

        E_g = symbol_values['E_g']

        if E_g > 0:
            is_metallic = False
        else:
            is_metallic = True

        return {
            'is_metallic': is_metallic
        }