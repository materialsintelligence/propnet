from propnet.core.models import AbstractModel
import sympy as sp

# TODO: this needs to handle conversion between *all* isotropic elastic moduli
# This is a design goal! No new models until we've figured this one out

class YoungsModulus(AbstractModel):

    @property
    def symbol_mapping(self):
        return {
            'E': 'youngs_modulus',
            'G': 'shear_modulus',
            'n': 'poisson_ratio',
            'K': 'bulk_modulus',
            'l': 'lame_first_parameter',
            'M': 'p_wave_modulus'
        }

    @property
    def valid_outputs(self):
        return ['E', 'G', 'n', 'K', 'l', 'M']

    def master_equations(self, values, output_symbol):

        # TODO: maybe symbol definitions should be handled by the base class?
        # unit wrapping will happen then anyway
        # perhaps need an AbstractAnalyticalModel base class?
        E, G, n, K, l, M = sp.symbols('E, G, n, K, l, M')

        equations = [
            ((3*K * (3*K - E)) / (9*K - E)) - l,
            ((3*K*E)/(9*K - E)) - G,
            ((3*K - E)/(6*K)) - n,
            ((3*K)*(3*K + E)/(9*K - E)) - M
        ]

        sp.linsolve(equations, ...)

    def assumptions(self):
        return [None] # change to Isotropic