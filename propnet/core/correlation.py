from maggma.builders import Builder
from itertools import combinations_with_replacement
import numpy as np
import json
from collections import defaultdict
from propnet.symbols import DEFAULT_SYMBOLS
from propnet import ureg
import logging
import re

logger = logging.getLogger(__name__)


class CorrelationBuilder(Builder):

    def __init__(self, propnet_store, mp_store,
                 correlation_store, out_file=None,
                 funcs='linlsq', **kwargs):

        self.propnet_store = propnet_store
        self.mp_store = mp_store
        self.correlation_store = correlation_store
        self.out_file = out_file

        self._correlation_funcs = {f.replace('_cfunc_', ''): getattr(self, f)
                                   for f in dir(self)
                                   if re.match(r'^_cfunc_.+$', f) and callable(getattr(self, f))}

        self._funcs = {}

        if isinstance(funcs, str):
            funcs = [funcs]

        for f in funcs:
            if isinstance(f, str) and f == 'all':
                self._funcs.update(self._correlation_funcs)
            elif isinstance(f, str) and f in self._correlation_funcs.keys():
                self._funcs[f] = self._correlation_funcs[f]
            elif callable(f):
                name = f.__module__ + "." + f.__name__
                self._funcs[name] = f
            else:
                raise ValueError("Invalid correlation function: {}".format(f))

        if not self._funcs:
            raise ValueError("No valid correlation functions selected")

        super(CorrelationBuilder, self).__init__(sources=[propnet_store, mp_store],
                                                 targets=[correlation_store],
                                                 **kwargs)

    def get_items(self):
        data = defaultdict(dict)
        propnet_props = [v.name for v in DEFAULT_SYMBOLS.values()
                         if (v.category == 'property' and v.shape == 1)]

        propnet_data = self.propnet_store.query(
            criteria={},
            properties=propnet_props + ['task_id', 'inputs'])

        for material in propnet_data:
            mpid = material['task_id']
            for prop, values in material.items():
                if prop in propnet_props:
                    data[mpid][prop] = ureg.Quantity(values['mean'], values['units'])
                elif prop == 'inputs':
                    input_d = defaultdict(list)
                    for q in values:
                        if q['symbol_type'] in propnet_props:
                            this_q = ureg.Quantity(q['value'], q['units'])
                            input_d[q['symbol_type']].append(this_q)
                    repeated_keys = set(input_d.keys()).intersection(set(data[mpid].keys()))
                    if repeated_keys:
                        print('Repeated key(s) from inputs: {}'.format(repeated_keys))
                    data[mpid].update(
                        {k: sum(v) / len(v) for k, v in input_d.items()})

        # TODO: Add these symbols to propnet so we don't have to bring them in explicitly?
        mp_query_props = ["piezo.eij_max", "elasticity.elastic_anisotropy", "elasticity.universal_anisotropy",
                          "diel.poly_electronic", "total_magnetization", "efermi",
                          "magnetism.total_magnetization_normalized_vol"]

        mp_props = [p.split(".")[1] if len(p.split(".")) == 2 else p for p in mp_query_props]

        mp_data = self.mp_store.query(
            criteria={},
            properties=mp_query_props + ['task_id']
        )

        for material in mp_data:
            mpid = material['task_id']
            for prop, value in material.items():
                if isinstance(value, dict):
                    for sub_prop, sub_value in value.items():
                        if prop + '.' + sub_prop in mp_query_props and sub_value:
                            data[mpid][sub_prop] = sub_value
                elif prop in mp_query_props and value:
                    data[mpid][prop] = value

        for prop_a, prop_b in combinations_with_replacement(propnet_props + mp_props, 2):
            x = []
            y = []
            for props_data in data.values():
                if prop_a in props_data.keys() and prop_b in props_data.keys():
                    x.append(props_data[prop_a])
                    y.append(props_data[prop_b])

            if x and any(isinstance(v, ureg.Quantity) for v in x):
                x_float = [xx.to(DEFAULT_SYMBOLS[prop_a].units).magnitude
                           if isinstance(xx, ureg.Quantity) else xx for xx in x]
            else:
                x_float = x
            if y and any(isinstance(v, ureg.Quantity) for v in y):
                y_float = [yy.to(DEFAULT_SYMBOLS[prop_b].units).magnitude
                           if isinstance(yy, ureg.Quantity) else yy for yy in y]
            else:
                y_float = y

            for name, func in self._funcs.items():
                data_dict = {'x_data': x_float,
                             'x_name': prop_a,
                             'y_data': y_float,
                             'y_name': prop_b,
                             'func': (name, func)}
                yield data_dict

    def process_item(self, item):
        prop_a, prop_b = item['x_name'], item['y_name']
        data_a, data_b = item['x_data'], item['y_data']
        func_name, func = item['func']
        n_points = len(data_a)

        if n_points < 2:
            correlation = 0.0
        else:
            correlation = func(data_a, data_b)
        return prop_a, prop_b, correlation, func_name, n_points

    @staticmethod
    def _cfunc_mic(x, y):
        from minepy import MINE
        m = MINE()
        m.compute_score(x, y)
        return m.mic()

    @staticmethod
    def _cfunc_linlsq(x, y):
        from scipy import stats
        fit = stats.linregress(x, y)
        return fit.rvalue ** 2

    @staticmethod
    def _cfunc_pearson(x, y):
        from scipy import stats
        fit = stats.pearsonr(x, y)
        return fit[0]

    @staticmethod
    def _cfunc_ransac(x, y):
        from sklearn.linear_model import RANSACRegressor
        r = RANSACRegressor(random_state=21)
        score = r.score(x[:, np.newaxis], y)
        return score

    @staticmethod
    def _cfunc_theilsen(x, y):
        from sklearn.linear_model import TheilSenRegressor
        r = TheilSenRegressor(random_state=21)
        score = r.score(x[:, np.newaxis], y)
        return score

    def update_targets(self, items):
        data = []
        for item in items:
            prop_a, prop_b, correlation, func_name, n_points = item
            data.append({'property_a': prop_a,
                         'property_b': prop_b,
                         'correlation': correlation,
                         'correlation_func': func_name,
                         'n_points': n_points,
                         'id': hash(prop_a) ^ hash(prop_b) ^ hash(func_name)})
        self.correlation_store.update(data, key='id')

    def finalize(self, cursor=None):
        if self.out_file:
            matrix = self.get_correlation_matrices()
            with open(self.out_file, 'w') as f:
                json.dump(matrix, f)

        super(CorrelationBuilder, self).finalize(cursor)

    def get_correlation_matrices(self, func_name=None):
        if not func_name:
            func_name = list(self._funcs.keys())

        prop_data = self.correlation_store.query(criteria={'property_a': {'$exists': True}},
                                                 properties=['property_a'])
        props = list(set(item['property_a'] for item in prop_data))

        out = {'properties': props,
               'n_points': None,
               'correlation': {}}

        for f in func_name:
            data = self.correlation_store.query(criteria={'correlation_func': f})
            corr_matrix: list = np.zeros(shape=(len(props), len(props))).tolist()

            fill_n_points = False
            if not out['n_points']:
                fill_n_points = True
                out['n_points'] = np.zeros(shape=(len(props), len(props))).tolist()

            for d in data:
                prop_a, prop_b, correlation, n_points = d['property_a'], \
                                                        d['property_b'], \
                                                        d['correlation'], \
                                                        d['n_points']
                ia, ib = props.index(prop_a), props.index(prop_b)
                corr_matrix[ia][ib] = correlation
                corr_matrix[ib][ia] = correlation

                if fill_n_points:
                    out['n_points'][ia][ib] = n_points
                    out['n_points'][ib][ia] = n_points

            out['correlation'][f] = corr_matrix

        return out

    def as_dict(self):
        d = super(CorrelationBuilder, self).as_dict()
        serialized_funcs = []
        for name, func in d['funcs'].items():
            if name in self._correlation_funcs.keys():
                serialized_funcs.append(name)
            else:
                logger.warning("Cannot serialize custom function '{}'. Omitting.".format(name))
        d['funcs'] = serialized_funcs
        return d
