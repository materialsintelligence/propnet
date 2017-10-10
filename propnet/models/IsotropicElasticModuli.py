from propnet.core.models import AbstractModel, validate_evaluate


class IsotropicElasticModuli(AbstractModel):

    # This is just a placeholder at present...

    @property
    def title(self):
        return "Convert between isotropic elastic moduli"

    @property
    def tags(self):
        return ["mechanical"]

    @property
    def description(self):
        return """
        This only supports one-way conversion for E, G and n
        at present. Working on an analyitcal version!
        """

    @property
    def references(self):
        return []

    @property
    def symbol_mapping(self):
        return {
            'E': 'youngs_modulus',
            'G': 'shear_modulus',
            'n': 'poisson_ratio',
            #'K': 'bulk_modulus',
            #'l': 'lame_first_parameter',
            #'M': 'p_wave_modulus'
        }

    @property
    def connections(self):
        return {
            'n': ('E', 'G'),
            'G': ('E', 'n'),
            'E': ('G', 'n')
        }

    @validate_evaluate
    def evaluate(self,
                 symbols_and_values_in,
                 symbol_out):

        # this is just placeholder for now, SymPy should do this for us
        # and if not, there's probably a less verbose way to define
        # this method

        if symbol_out == 'E':

            G = symbols_and_values_in.get('G')
            n = symbols_and_values_in.get('n')

            return 2*G*(1+n)

        elif symbol_out == 'G':

            E = symbols_and_values_in.get('E')
            n = symbols_and_values_in.get('n')

            return E / (2*(1+n))

        elif symbol_out == 'n':

            G = symbols_and_values_in.get('G')
            E = symbols_and_values_in.get('E')

            return (E/(2*G)) - 1

    @property
    def test_sets(self):
        return [
            {'K': (0, "GPa"),
             'E': (0, "GPa"),
             'G': (0, "GPa")}
        ]

    #def equations(self, E, G, n, K, l, M):
    #    return [
    #        ((3 * K * (3 * K - E)) / (9 * K - E)) - l,
    #        ((3 * K * E) / (9 * K - E)) - G,
    #        ((3 * K - E) / (6 * K)) - n,
    #        ((3 * K) * (3 * K + E) / (9 * K - E)) - M
    #    ]