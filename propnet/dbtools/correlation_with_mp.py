"""
WARNING: THIS MODULE IS DEPRECATED AND WILL BE REMOVED IN THE NEAR FUTURE.
This module is only here for backwards compatibility. Use ``propnet.dbtools.correlation`` instead.
"""
import json
import logging
import re
from itertools import product
from collections import defaultdict
import warnings

from maggma.builders import Builder
import numpy as np

from propnet.core.graph import Graph
from propnet import ureg
# noinspection PyUnresolvedReferences
import propnet.models
from propnet.core.registry import Registry

warnings.warn("The correlation_with_mp module is deprecated. Use the correlation module instead.",
              DeprecationWarning)
logger = logging.getLogger(__name__)


class CorrelationBuilder(Builder):
    """
    A class to calculate the correlation between properties derived by or used in propnet
    using a suite of regression tools. Uses the Builder architecture for optional parallel
    processing of data.

    Note: serialization of builder does not work with custom correlation functions, although
    interactive use does support them.

    """
    # TODO: Add these symbols to propnet so we don't have to bring them in explicitly?
    MP_QUERY_PROPS = ["piezo.eij_max", "elasticity.universal_anisotropy",
                      "diel.poly_electronic", "total_magnetization", "efermi",
                      "magnetism.total_magnetization_normalized_vol"]
    PROPNET_PROPS = [v.name for v in Registry("symbols").values()
                     if (v.category == 'property' and v.shape == 1)]

    def __init__(self, propnet_store, mp_store,
                 correlation_store, out_file=None,
                 funcs='linlsq', props=None, **kwargs):
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
                spearman: Spearman rank correlation, reports r
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

        mp_prop_map = {(p.split(".")[1] if len(p.split(".")) == 2 else p): p for p in self.MP_QUERY_PROPS}
        self._props = props
        if not props:
            self.mp_query_props = self.MP_QUERY_PROPS
            self.mp_props = list(mp_prop_map.keys())
            self.propnet_props = self.PROPNET_PROPS
        else:
            self.propnet_props = []
            self.mp_props = []
            self.mp_query_props = []
            if isinstance(props, str):
                props = [props]
            for p in props:
                if p in self.PROPNET_PROPS:
                    self.propnet_props.append(p)
                elif p in mp_prop_map.keys():
                    self.mp_props.append(p)
                    self.mp_query_props.append(mp_prop_map[p])

        super(CorrelationBuilder, self).__init__(sources=[propnet_store, mp_store],
                                                 targets=[correlation_store],
                                                 **kwargs)

    def get_items(self):
        """
        Collects scalar data from propnet and MP databases, aggregates it by property, and creates
        a generator to iterate over all pairs of properties, including pairing of the same property
        with itself for sanity check, and correlation functions.

        Returns: (generator) a generator providing a dictionary with the data for correlation:
            {'x_data': (list<float>) data for independent property (x-axis),
             'x_name': (str) name of independent property,
             'y_data': (list<float>) data for dependent property (y-axis),
             'y_name': (str) name of dependent property,
             'func': (tuple<str, function>) name and function handle for correlation function
             }

        """
        data = defaultdict(dict)

        propnet_data = self.propnet_store.query(
            criteria={},
            properties=[p + '.mean' for p in self.propnet_props] +
                       [p + '.units' for p in self.propnet_props] +
                       [p + '.quantities' for p in self.propnet_props] +
                       ['task_id', 'inputs'])

        for material in propnet_data:
            mpid = material['task_id']

            input_d = defaultdict(list)
            for q in material['inputs']:
                if q['symbol_type'] in self.propnet_props:
                    this_q = ureg.Quantity(q['value'], q['units'])
                    input_d[q['symbol_type']].append(this_q)

            for prop, values in material.items():
                if prop in self.propnet_props:
                    if prop in input_d.keys():
                        for q in values['quantities']:
                            input_d[prop].append(ureg.Quantity(q['value'], q['units']))
                    else:
                        this_q = ureg.Quantity(values['mean'], values['units'])
                        input_d[prop] = [this_q]

            data[mpid].update(
                {k: sum(v) / len(v) for k, v in input_d.items()})

        # TODO: Add these symbols to propnet so we don't have to bring them in explicitly?

        mp_data = self.mp_store.query(
            criteria={},
            properties=self.mp_query_props + ['task_id']
        )

        for material in mp_data:
            mpid = material['task_id']
            for prop, value in material.items():
                if isinstance(value, dict):
                    for sub_prop, sub_value in value.items():
                        if prop + '.' + sub_prop in self.mp_query_props and sub_value is not None:
                            data[mpid][sub_prop] = sub_value
                elif prop in self.mp_query_props and value is not None:
                    data[mpid][prop] = value

        # product() produces all possible combinations of properties
        for prop_x, prop_y in product(self.propnet_props + self.mp_props, repeat=2):
            x = []
            y = []
            for props_data in data.values():
                if prop_x in props_data.keys() and prop_y in props_data.keys():
                    x.append(props_data[prop_x])
                    y.append(props_data[prop_y])

            # MP data does not have units listed in database, so will be floats. propnet
            # data may not have the same units as the MP data, so is stored as pint
            # quantities. Here, the quantities are coerced into the units of MP data
            # as stored in symbols and coverts them to floats.
            if x and any(isinstance(v, ureg.Quantity) for v in x):
                x_float = [xx.to(Registry("symbols")[prop_x].units).magnitude
                           if isinstance(xx, ureg.Quantity) else xx for xx in x]
            else:
                x_float = x
            if y and any(isinstance(v, ureg.Quantity) for v in y):
                y_float = [yy.to(Registry("symbols")[prop_y].units).magnitude
                           if isinstance(yy, ureg.Quantity) else yy for yy in y]
            else:
                y_float = y

            for name, func in self._funcs.items():
                data_dict = {'x_data': x_float,
                             'x_name': prop_x,
                             'y_data': y_float,
                             'y_name': prop_y,
                             'func': (name, func)}
                yield data_dict

    def process_item(self, item):
        """
        Run correlation calculation on a pair of properties using the specified function.

        Args:
            item: (dict) input provided by get_items() (see get_items() for structure)

        Returns: (tuple<str, str, float, str, int>) output of calculation with necessary
            information about calculation included. Format in tuple:
                independent property (x-axis) name,
                dependent property (y-axis) name,
                correlation value,
                correlation function name,
                number of data points used for correlation
                length of shortest path between properties on propnet graph where x-axis property
                    is starting property and y-axis property is ending property.
                    Note: if no (forward) connection exists, the path length will be None. This does
                    not preclude y->x having a forward path.

        """
        prop_x, prop_y = item['x_name'], item['y_name']
        data_x, data_y = item['x_data'], item['y_data']
        func_name, func = item['func']
        n_points = len(data_x)

        g = Graph()
        try:
            path_length = g.get_degree_of_separation(prop_x, prop_y)
        except ValueError:
            path_length = None

        if n_points < 2:
            correlation = 0.0
        else:
            correlation = func(data_x, data_y)
        return prop_x, prop_y, correlation, func_name, n_points, path_length

    @staticmethod
    def _cfunc_mic(x, y):
        """
        Get maximal information coefficient for data set.

        Args:
            x: (list<float>) independent property (x-axis)
            y: (list<float>) dependent property (y-axis)

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
            x: (list<float>) independent property (x-axis)
            y: (list<float>) dependent property (y-axis)

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
            x: (list<float>) independent property (x-axis)
            y: (list<float>) dependent property (y-axis)

        Returns: (float) Pearson R value

        """
        from scipy import stats
        fit = stats.pearsonr(x, y)
        return fit[0]

    @staticmethod
    def _cfunc_spearman(x, y):
        """
        Get R value for Spearman fit of a data set.

        Args:
            x: (list<float>) independent property (x-axis)
            y: (list<float>) dependent property (y-axis)

        Returns: (float) Spearman R value

        """
        from scipy import stats
        fit = stats.spearmanr(x, y)
        return fit[0]

    @staticmethod
    def _cfunc_ransac(x, y):
        """
        Get random sample consensus (RANSAC) regression score for data set.

        Args:
            x: (list<float>) independent property (x-axis)
            y: (list<float>) dependent property (y-axis)

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
            x: (list<float>) independent property (x-axis)
            y: (list<float>) dependent property (y-axis)

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
            prop_x, prop_y, correlation, func_name, n_points, path_length = item
            data.append({'property_x': prop_x,
                         'property_y': prop_y,
                         'correlation': correlation,
                         'correlation_func': func_name,
                         'n_points': n_points,
                         'shortest_path_length': path_length,
                         'id': hash((prop_x, prop_y)) ^ hash(func_name)})
        self.correlation_store.update(data, key='id')

    def finalize(self, cursor=None):
        """
        Outputs correlation data to JSON file, if specified in instantiation, and runs
        clean-up function for Builder.

        Args:
            cursor: (Mongo Store cursor) optional, cursor to close if not automatically closed.

        """
        if self.out_file:
            try:
                self.write_correlation_data_file(self.out_file)
            except OSError:
                logger.warning("Cannot open file for writing! Skipping file writing.")

        super(CorrelationBuilder, self).finalize(cursor)

    def write_correlation_data_file(self, out_file):
        """
        Gets data dictionary containing correlation matrices and outputs to a file.

        Args:
            out_file: (str) file path and name for output to JSON file
        """
        matrix = self.get_correlation_matrices()
        with open(out_file, 'w') as f:
            json.dump(matrix, f)

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

        prop_data = self.correlation_store.query(criteria={'property_x': {'$exists': True}},
                                                 properties=['property_x'])
        props = list(set(item['property_x'] for item in prop_data))

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
                prop_x, prop_y, correlation, n_points, path_length = d['property_x'], \
                                                                     d['property_y'], \
                                                                     d['correlation'], \
                                                                     d['n_points'], \
                                                                     d['shortest_path_length']
                ia, ib = props.index(prop_x), props.index(prop_y)
                corr_matrix[ia][ib] = correlation

                if fill_info_matrices:
                    out['n_points'][ia][ib] = n_points
                    out['n_points'][ib][ia] = n_points
                    out['shortest_path_length'][ia][ib] = path_length

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