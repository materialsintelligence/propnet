from maggma.builders import Builder
from itertools import combinations_with_replacement
import numpy as np
import json
from minepy import MINE
from collections import defaultdict
from propnet.symbols import DEFAULT_SYMBOLS
from propnet import ureg

import random

class MicBuilder(Builder):

    def __init__(self, propnet_store, mp_store, correlation_store, out_file, **kwargs):

        self.propnet_store = propnet_store
        self.mp_store = mp_store
        self.correlation_store = correlation_store
        self.out_file = out_file

        super(MicBuilder, self).__init__(sources=[propnet_store, mp_store],
                                         targets=[correlation_store],
                                         **kwargs)

    def get_items(self):
        data = defaultdict(dict)
        propnet_props = [v.name for v in DEFAULT_SYMBOLS.values()
                         if (v.category == 'property' and v.shape == 1)]

        # Debug with subset of mpids
        # mpids = self.propnet_store.query(criteria={}, properties=['task_id'])
        # mpids = [m['task_id'] for m in mpids]
        # count = 0
        # mpids_to_process = []
        # while count < 1000:
        #     idx = random.randint(0, len(mpids)-1)
        #     mpids_to_process.append(mpids.pop(idx))
        #     count += 1
        #
        # propnet_data = self.propnet_store.query(
        #     criteria={'task_id': {'$in': mpids_to_process}},
        #     properties=propnet_props.copy().extend(['task_id', 'inputs']))

        propnet_data = self.propnet_store.query(
            criteria={},
            properties=propnet_props.copy().extend(['task_id', 'inputs']))

        for material in propnet_data:
            for prop, values in material.items():
                if prop in propnet_props:
                    data[material['task_id']][prop] = values['mean']
                elif prop == 'inputs':
                    input_d = defaultdict(list)
                    for q in values:
                        if q['symbol_type'] in propnet_props:
                            this_q = ureg.Quantity(q['value'], q['units'])
                            input_d[q['symbol_type']].append(this_q)
                    data[material['task_id']].update(
                        {k: sum(v) / len(v) for k, v in input_d.items()})

        mp_props = ["eij_max", "elastic_anisotropy", "universal_anisotropy",
                    "poly_electronic", "total_magnetization", "efermi",
                    "total_magnetization_normalized_vol"]

        # mp_data = self.mp_store.query(
        #     criteria={'task_id': {'$in': mpids_to_process}},
        #     properties=mp_props.copy().extend('task_id')
        # )

        mp_data = self.mp_store.query(
            criteria={},
            properties=mp_props.copy().extend('task_id')
        )

        for material in mp_data:
            for prop, value in material.items():
                if prop in mp_props:
                    data[material['task_id']][prop] = value

        for prop_a, prop_b in combinations_with_replacement(propnet_props + mp_props, 2):
            x = []
            y = []
            for props_data in data.values():
                if prop_a in props_data.keys() and prop_b in props_data.keys():
                    x.append(props_data[prop_a])
                    y.append(props_data[prop_b])

            if prop_a in propnet_props:
                x = [xx.to(DEFAULT_SYMBOLS[prop_a].units).magnitude if isinstance(xx, ureg.Quantity) else xx for xx in x]
            if prop_b in propnet_props:
                y = [yy.to(DEFAULT_SYMBOLS[prop_b].units).magnitude if isinstance(yy, ureg.Quantity) else yy for yy in y]

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
        self.correlation_store.update(data, key='id')

    def finalize(self, cursor=None):
        if self.out_file:
            matrix = self.get_properties_matrix()
            with open(self.out_file, 'w') as f:
                json.dump(matrix, f)

        super(MicBuilder, self).finalize(cursor)

    def get_properties_matrix(self):
        data = self.correlation_store.query(criteria={'property_a': {'$exists': True}})
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
