from maggma.builders import Builder
from itertools import combinations_with_replacement
import numpy as np
import json
from collections import defaultdict
from propnet.symbols import DEFAULT_SYMBOLS
from propnet.core.graph import Graph
from propnet import ureg
import logging
import re

logger = logging.getLogger(__name__)


class CorrelationBuilder(Builder):
    """
    A class to calculate the correlation between properties derived by or used in propnet
    using a suite of regression tools. Uses the Builder architecture for optional parallel
    processing of data.

    Note: serialization of builder does not work with custom correlation functions, although
    interactive use does support them.

    """
    def __init__(self, propnet_store, mp_store,
                 correlation_store, out_file=None,
                 funcs='linlsq', **kwargs):
        """
        Constructor for the correlation builder.

        Args:
            propnet_store: (Mongolike Store) store instance pointing to propnet collection
                with read access
            mp_store: (Mongolike Store) store instance pointing to Materials Project collection with read access
            correlation_store: (Mongolike Store) store instance pointing to collection with write access
            out_file: (str) optional, filename to output data in JSON format (useful if using a MemoryStore
                for correlation_store)
            funcs: (str, function, list<str, function>) functions to use for correlation. Built-in functions can
                be specified by the following strings:

                linlsq (default): linear least-squares, reports R^2
                pearson: Pearson r-correlation, reports r
                mic: maximal-information non-parametric exploration, reports maximal information coefficient
                ransac: random sample consensus (RANSAC) regression, reports score
                theilsen: Theil-Sen regression, reports score
                all: runs all correlation functions above
            **kwargs: arguments to the Builder superclass
        """

        self.propnet_store = propnet_store
        self.mp_store = mp_store
        self.correlation_store = correlation_store
        self.out_file = out_file

        self._correlation_funcs = {f.replace('_cfunc_', ''): getattr(self, f)
                                   for f in dir(self)
                                   if re.match(r'^_cfunc_.+$', f) and callable(getattr(self, f))}

        self._funcs = {}

        if not isinstance(funcs, list):
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
        """
        Collects scalar data from propnet and MP databases, aggregates it by property, and creates
        a generator to iterate over all pairs of properties, including pairing of the same property
        with itself for sanity check, and correlation functions.

        Returns: (generator) a generator providing a dictionary with the data for correlation:
            {'x_data': (list<float>) data for first property (x-axis),
             'x_name': (str) name of first property,
             'y_data': (list<float>) data for second property (y-axis),
             'y_name': (str) name of second property,
             'func': (tuple<str, function>) name and function handle for correlation function
             }

        """
        data = defaultdict(dict)
        propnet_props = [v.name for v in DEFAULT_SYMBOLS.values()
                         if (v.category == 'property' and v.shape == 1)]

        propnet_data = self.propnet_store.query(
            criteria={},
            properties=[p + '.mean' for p in propnet_props] +
                       [p + '.units' for p in propnet_props] +
                       ['task_id', 'inputs'])

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
                        logger.warning('Repeated key(s) from inputs: {}'.format(repeated_keys))
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

            # MP data does not have units listed in database, so will be floats. propnet
            # data may not have the same units as the MP data, so is stored as pint
            # quantities. Here, the quantities are coerced into the units of MP data
            # as stored in symbols and coverts them to floats.
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
        """
        Run correlation calculation on a pair of properties using the specified function.

        Args:
            item: (dict) input provided by get_items() (see get_items() for structure)

        Returns: (tuple<str, str, float, str, int>) output of calculation with necessary
            information about calculation included. Format in tuple:
                property A name,
                property B name,
                correlation value,
                correlation function name,
                number of data points used for correlation
                length of shortest path between properties on propnet graph (-1 if not connected)

        """
        prop_a, prop_b = item['x_name'], item['y_name']
        data_a, data_b = item['x_data'], item['y_data']
        func_name, func = item['func']
        n_points = len(data_a)

        g = Graph()
        try:
            path_lengths = [g.get_degree_of_separation(prop_a, prop_b),
                            g.get_degree_of_separation(prop_b, prop_a)]
            path_lengths = [p for p in path_lengths if p is not None]
            if path_lengths:
                path_length = min(path_lengths)
            else:
                path_length = None
        except ValueError:
            path_length = None

        if n_points < 2:
            correlation = 0.0
        else:
            correlation = func(data_a, data_b)
        return prop_a, prop_b, correlation, func_name, n_points, path_length

    @staticmethod
    def _cfunc_mic(x, y):
        """
        Get maximal information coefficient for data set.

        Args:
            x: (list<float>) property A
            y: (list<float>) property B

        Returns: (float) maximal information coefficient

        """
        from minepy import MINE
        m = MINE()
        m.compute_score(x, y)
        return m.mic()

    @staticmethod
    def _cfunc_linlsq(x, y):
        """
        Get R^2 value for linear least-squares fit of a data set.

        Args:
            x: (list<float>) property A
            y: (list<float>) property B

        Returns: (float) R^2 value

        """
        from scipy import stats
        fit = stats.linregress(x, y)
        return fit.rvalue ** 2

    @staticmethod
    def _cfunc_pearson(x, y):
        """
        Get R value for Pearson fit of a data set.

        Args:
            x: (list<float>) property A
            y: (list<float>) property B

        Returns: (float) Pearson R value

        """
        from scipy import stats
        fit = stats.pearsonr(x, y)
        return fit[0]

    @staticmethod
    def _cfunc_ransac(x, y):
        """
        Get random sample consensus (RANSAC) regression score for data set.

        Args:
            x: (list<float>) property A
            y: (list<float>) property B

        Returns: (float) RANSAC score

        """
        from sklearn.linear_model import RANSACRegressor
        r = RANSACRegressor(random_state=21)
        x_coeff = np.array(x)[:, np.newaxis]
        r.fit(x_coeff, y)
        return r.score(x_coeff, y)

    @staticmethod
    def _cfunc_theilsen(x, y):
        """
        Get Theil-Sen regression score for data set.

        Args:
            x: (list<float>) property A
            y: (list<float>) property B

        Returns: (float) Theil-Sen score

        """
        from sklearn.linear_model import TheilSenRegressor
        r = TheilSenRegressor(random_state=21)
        x_coeff = np.array(x)[:, np.newaxis]
        r.fit(x_coeff, y)
        return r.score(x_coeff, y)

    def update_targets(self, items):
        """
        Write correlation data to Mongo store.

        Args:
            items: (list<dict>) list of results output by process_item()

        """
        data = []
        for item in items:
            prop_a, prop_b, correlation, func_name, n_points, path_length = item
            # This is so the hash is the same if prop_a and prop_b are swapped
            sorted_props = [prop_a, prop_b]
            sorted_props.sort()
            data.append({'property_a': prop_a,
                         'property_b': prop_b,
                         'correlation': correlation,
                         'correlation_func': func_name,
                         'n_points': n_points,
                         'shortest_path_length': path_length,
                         'id': hash(sorted_props[0]) ^ hash(sorted_props[1]) ^ hash(func_name)})
        self.correlation_store.update(data, key='id')

    def finalize(self, cursor=None):
        """
        Outputs correlation data to JSON file, if specified in instantiation, and runs
        clean-up function for Builder.

        Args:
            cursor: (Mongo Store cursor) optional, cursor to close if not automatically closed.

        """
        if self.out_file:
            matrix = self.get_correlation_matrices()
            with open(self.out_file, 'w') as f:
                json.dump(matrix, f)

        super(CorrelationBuilder, self).finalize(cursor)

    def get_correlation_matrices(self, func_name=None):
        """
        Builds document containing the correlation matrix with relevant data regarding
        correlation algorithm and properties of the data set.

        Args:
            func_name: (str) optional, name of the correlation functions to include in the document
                default: None, which is to include all that were run by this builder.

        Returns: (dict) document containing correlation data. Format:
            {'properties': (list<str>) names of properties calculated in order of how they are indexed
                    in the matrices
             'n_points': (list<list<int>>) list of lists (i.e. matrix) containing the number of data
                    points evaluated during the fitting procedure
             'correlation': (dict<str: list<list<float>>>) dictionary of matrices containing correlation
                    results, keyed by correlation function name
            }

        """

        prop_data = self.correlation_store.query(criteria={'property_a': {'$exists': True}},
                                                 properties=['property_a'])
        props = list(set(item['property_a'] for item in prop_data))

        out = {'properties': props,
               'n_points': None,
               'shortest_path_length': None,
               'correlation': {}}

        if not func_name:
            func_name = list(self._funcs.keys())

        if isinstance(func_name, str):
            func_name = [func_name]

        for f in func_name:
            data = self.correlation_store.query(criteria={'correlation_func': f})
            corr_matrix: list = np.zeros(shape=(len(props), len(props))).tolist()

            fill_info_matrices = False
            if not out['n_points'] and not out['shortest_path_length']:
                fill_info_matrices = True
                out['n_points'] = np.zeros(shape=(len(props), len(props))).tolist()
                out['shortest_path_length'] = np.zeros(shape=(len(props), len(props))).tolist()

            for d in data:
                prop_a, prop_b, correlation, n_points, path_length = d['property_a'], \
                                                                     d['property_b'], \
                                                                     d['correlation'], \
                                                                     d['n_points'], \
                                                                     d['shortest_path_length']
                ia, ib = props.index(prop_a), props.index(prop_b)
                corr_matrix[ia][ib] = correlation
                corr_matrix[ib][ia] = correlation

                if fill_info_matrices:
                    out['n_points'][ia][ib] = n_points
                    out['n_points'][ib][ia] = n_points
                    out['shortest_path_length'][ia][ib] = path_length
                    out['shortest_path_length'][ib][ia] = path_length

            out['correlation'][f] = corr_matrix

        return out

    def as_dict(self):
        """
        Returns the representation of the builder as a dictionary in JSON serializable format.
        Note: because functions are not JSON serializable, custom functions are omitted when
            serializing the object.

        Returns: (dict) representation of this builder as a JSON-serializable dictionary

        """
        d = super(CorrelationBuilder, self).as_dict()
        serialized_funcs = []
        for name in d['funcs'].keys():
            if name in self._correlation_funcs.keys():
                serialized_funcs.append(name)
            else:
                logger.warning("Cannot serialize custom function '{}'. Omitting.".format(name))

        if not serialized_funcs:
            logger.warning("No functions were able to be serialized from this builder.")

        d['funcs'] = serialized_funcs
        return d
