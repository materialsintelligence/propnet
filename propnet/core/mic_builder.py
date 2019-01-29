from maggma.builders import Builder
from itertools import combinations_with_replacement
import numpy as np
import json
from minepy import MINE


class MicBuilder(Builder):

    def __init__(self, data_store, covariance_store, out_file, **kwargs):

        self.data_store = data_store
        self.covariance_store = covariance_store
        self.out_file = out_file

        super(MicBuilder, self).__init__(sources=[data_store],
                                         targets=[covariance_store],
                                         **kwargs)

    def get_items(self):
        props = ["sound_velocity_transverse", "sound_velocity_longitudinal", "debye_temperature",
                 "ionic_radius_b", "volume_unit_cell", "lame_first_parameter", "shear_modulus",
                 "thermal_conductivity", "pugh_ratio", "poisson_ratio", "hhi_production",
                 "band_gap_gw", "melting_point", "relative_permeability",
                 "dielectric_figure_of_merit_current_leakage", "nsites", "molar_mass",
                 "vickers_hardness", "cost_per_kg", "hhi_reserve", "density", "refractive_index",
                 "dielectric_figure_of_merit_energy", "atomic_density", "interatomic_spacing",
                 "ionic_radius_a", "mass_per_atom", "p_wave_modulus", "band_gap_pbe", "cost_per_mol",
                 "bulk_modulus", "sound_velocity_mean", "volume_per_atom", "gruneisen_parameter",
                 "youngs_modulus", "band_gap"]
        data = self.data_store.query(
            criteria={},
            properties=[p + '.mean' for p in props])
        data = [d for d in data]

        for prop_a, prop_b in combinations_with_replacement(props, 2):
            x = []
            y = []
            for d in data:
                if prop_a in d.keys() and prop_b in d.keys():
                    x.append(d[prop_a]['mean'])
                    y.append(d[prop_b]['mean'])

            data_dict = {prop_a: x,
                         prop_b: y}
            yield data_dict

    def process_item(self, item):
        if len(item.keys()) == 1:
            prop_a = list(item.keys())[0]
            data_a = item[prop_a]
            prop_b = prop_a
            data_b = data_a
        else:
            prop_a, prop_b = list(item.keys())
            data_a, data_b = item[prop_a], item[prop_b]

        m = MINE()
        m.compute_score(data_a, data_b)
        return prop_a, prop_b, m.mic()

    def update_targets(self, items):
        data = []
        for item in items:
            prop_a, prop_b, covariance = item
            data.append({'property_a': prop_a,
                         'property_b': prop_b,
                         'covariance': covariance,
                         'id': hash(prop_a) ^ hash(prop_b)})
        self.covariance_store.update(data, key='id')

    def finalize(self, cursor=None):
        if self.out_file:
            matrix = self.get_properties_matrix()
            with open(self.out_file, 'w') as f:
                json.dump(matrix, f)

        super(MicBuilder, self).finalize(cursor)

    def get_properties_matrix(self):
        data = self.covariance_store.query(criteria={'property_a': {'$exists': True}})
        data = [d for d in data]
        props = list(set(d['property_a'] for d in data))
        matrix = np.zeros(shape=(len(props), len(props))).tolist()

        for d in data:
            prop_a, prop_b, covariance = d['property_a'], d['property_b'], d['covariance']
            ia, ib = props.index(prop_a), props.index(prop_b)
            matrix[ia][ib] = covariance
            matrix[ib][ia] = covariance

        return {'properties': props,
                'matrix': matrix}
