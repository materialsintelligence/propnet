from propnet.core.models import AbstractModel

from gbml.elasticity import predict_k_g_from_entry


class Gbml(AbstractModel):

    def plug_in(self, symbol_values):

        computed_entry = symbol_values['computed_entry']

        entry = {
            'material_id': 'mp',
            'energy_per_atom': computed_entry.energy_per_atom,
            'is_hubbard': computed_entry.parameters['is_hubbard'],
            'nsites': symbol_values['nsites'],
            'pretty_formula': symbol_values['formula'],
            'volume': symbol_values['volume']
        }

        K, G, caveats = predict_k_g_from_entry(entry)

        return {
            'K': K,
            'G': G
        }
