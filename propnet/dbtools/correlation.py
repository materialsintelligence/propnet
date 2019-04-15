from maggma.builders import Builder
from itertools import combinations_with_replacement
import numpy as np
import json
from collections import defaultdict
from propnet.core.graph import Graph
from propnet import ureg
import logging
import re

# noinspection PyUnresolvedReferences
import propnet.models
from propnet.core.registry import Registry

logger = logging.getLogger(__name__)


class CorrelationBuilder(Builder):
    """
    A class to calculate the correlation between properties derived by or used in propnet
    using a suite of regression tools. Uses the Builder architecture for optional parallel
    processing of data.

    Note: serialization of builder does not work with custom correlation functions, although
    interactive use does support them.

    """
    PROPNET_PROPS = [v.name for v in Registry("symbols").values()
                     if (v.category == 'property' and v.shape == 1)]

    def __init__(self, propnet_store,
                 correlation_store, out_file=None,
                 funcs='linlsq', props=None,
                 sample_size=None, **kwargs):
        """
        Constructor for the correlation builder.

        Args:
            propnet_store (Mongolike Store): store instance pointing to propnet collection
                with read access
            correlation_store (Mongolike Store): store instance pointing to collection with write access
            out_file (str): optional, filename to output data in JSON format (useful if using a MemoryStore
                for correlation_store)
            funcs (`str`, `callable`, list of `str` or `callable`) functions to use for correlation.
                Built-in functions can be specified by the following strings:

                linlsq (default): linear least-squares, reports R^2
                pearson: Pearson r-correlation, reports r
                spearman: Spearman rank correlation, reports r
                mic: maximal-information non-parametric exploration, reports maximal information coefficient
                ransac: random sample consensus (RANSAC) regression, reports score
                theilsen: Theil-Sen regression, reports score
                all: runs all correlation functions above
            props (`list` of `str`): optional, list of properties for which to calculate the correlation.
                Default is to calculate for all possible pairs (props=None)
            sample_size (int): optional, limits correlation calculation data to a random sample of size
                `sample_size`. Default: None (no limit)
            **kwargs: arguments to the Builder superclass
        """

        self.propnet_store = propnet_store
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

        self._props = props or self.PROPNET_PROPS
        if sample_size is not None and sample_size < 2:
            raise ValueError("Sample size must be greater than 1")
        self.sample_size = sample_size
        self.total = len(self._props)**2 * len(self._funcs)

        super(CorrelationBuilder, self).__init__(sources=[propnet_store],
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
        # Update total in case we changed something between instantiation and running
        self.total = len(self._props)**2 * len(self._funcs)

        # combinations_with_replacement() produces all possible pairs of properties
        # without repeating, i.e. will give AB but not BA. Code below manually
        # produces "BA" so that we don't have to re-query the database.
        for prop_x, prop_y in combinations_with_replacement(self._props, 2):
            # Get all materials which have both properties in the inputs or outputs
            criteria = {
                    '$and': [
                        {'$or': [
                            {'inputs.symbol_type': prop_x},
                            {prop_x: {'$exists': True}}]},
                        {'$or': [
                            {'inputs.symbol_type': prop_y},
                            {prop_y: {'$exists': True}}]}
                    ]}
            properties = [prop_x + '.quantities', prop_y + '.quantities', 'inputs']

            if self.sample_size is None:
                pn_data = self.propnet_store.query(criteria=criteria,
                                                   properties=properties)
            else:
                pipeline = [
                    {'$match': criteria},
                    {'$sample': {'size': self.sample_size}},
                    {'$project': {p: True for p in properties}},
                ]
                pn_data = self.propnet_store.collection.aggregate(
                    pipeline, allowDiskUse=True
                )

            x_unit = Registry("units")[prop_x]
            y_unit = Registry("units")[prop_y]
            data = defaultdict(list)
            for material in pn_data:
                # Collect all data with units for this material
                # and calculate the mean, convert units, store magnitude of mean
                if prop_x == prop_y:
                    # This is to avoid duplicating the work and the data
                    props = (prop_x,)
                    units = (x_unit,)
                else:
                    props = (prop_x, prop_y)
                    units = (x_unit, y_unit)
                for prop, unit in zip(props, units):
                    qs = [ureg.Quantity(q['value'], q['units'])
                          for q in material['inputs']
                          if q['symbol_type'] == prop]
                    if prop in material:
                        qs.extend([ureg.Quantity(q['value'], q['units'])
                                   for q in material[prop]['quantities']])

                    if len(qs) == 0:
                        raise ValueError("Query for property {} gave no results"
                                         "".format(prop))
                    prop_mean = sum(qs) / len(qs)
                    data[prop].append(prop_mean.to(unit).magnitude)

            # So we get AB and BA without re-querying, but not two AA
            if prop_x == prop_y:
                prop_combos = ((prop_x, prop_x),)
            else:
                prop_combos = ((prop_x, prop_y), (prop_y, prop_x))
            for x, y in prop_combos:
                for name, func in self._funcs.items():
                    data_dict = {'x_data': data[x],
                                 'x_name': x,
                                 'y_data': data[y],
                                 'y_name': y,
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
            path_length_xy = g.get_degree_of_separation(prop_x, prop_y)
            path_length_yx = g.get_degree_of_separation(prop_y, prop_x)
        except ValueError:
            # This shouldn't happen...but just in case
            path_length_xy = None
            path_length_yx = None

        try:
            path_length = min(path_length_xy, path_length_yx)
        except TypeError:
            path_length = path_length_xy or path_length_yx

        if n_points < 2:
            result = 0.0
        else:
            try:
                result = func(data_x, data_y)
            except Exception as ex:
                # If correlation fails, catch the error, save it, and move on
                result = ex
        return prop_x, prop_y, result, func_name, n_points, path_length

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
            prop_x, prop_y, result, func_name, n_points, path_length = item
            d = {'property_x': prop_x,
                 'property_y': prop_y,
                 'correlation_func': func_name,
                 'n_points': n_points,
                 'shortest_path_length': path_length,
                 'id': hash((prop_x, prop_y)) ^ hash(func_name)}
            if not isinstance(result, Exception):
                d['correlation'] = result
            else:
                d['correlation'] = None
                d['error'] = (result.__class__.__name__,
                              result.args)
            data.append(d)
        self.correlation_store.update(data, key='id')

    def finalize(self, cursor=None):
        """
        Outputs correlation data to JSON file, if specified in instantiation, and runs
        clean-up function for Builder.

        Args:
            cursor: (Mongo Store cursor) optional, cursor to close if not automatically closed.

        """

        props_to_index = ['property_x', 'property_y', 'correlation_func']
        for prop in props_to_index:
            if not self.correlation_store.ensure_index(prop):
                logger.warning("Could not add index for property {}".format(prop))

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
