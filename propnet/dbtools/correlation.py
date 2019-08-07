"""Database builder for correlation calculations.

This module calculates correlation scores between scalar materials properties
to elucidate relationships between properties and writes the results to a MongoDB
collection with optional output to a file. The builder requires a propnet
quantity database (preferred) or a full materials database (slower) as input.

Example:
    The builder can be executed using a ``maggma`` Runner in a Python script:

    >>> from propnet.dbtools.correlation import CorrelationBuilder
    >>> from maggma.stores import MongoStore
    >>> from maggma.runner import Runner
    >>> pn_quantity_db = MongoStore(...)    # read access
    >>> pn_correlation_db = MongoStore(...) # write access
    >>> cb = CorrelationBuilder(pn_quantity_db, pn_correlation_db,
    >>>                         from_quantity_db=True)
    >>> runner = Runner([cb])
    >>> runner.run()

    It can also be run using the ``maggma`` command line tool. In Python, build
    the Runner as above, and then:

    >>> from monty.serialization import dumpfn
    >>> dumpfn(runner, 'correlation_runner.json')

    And then in the terminal::

        $ mrun -n NUM_PROCS correlation_runner.json

    If running on a large data set (>100k materials), ``NUM_PROCS`` should be small
    to ensure the system RAM is not used up, especially if using ``'mic'`` correlation.

"""

import logging
import re
import json
from collections import defaultdict
from itertools import combinations_with_replacement

import numpy as np
from maggma.builders import Builder

from propnet.core.graph import Graph
from propnet import ureg
# noinspection PyUnresolvedReferences
import propnet.models
from propnet.core.registry import Registry

logger = logging.getLogger(__name__)
"""logging.Logger: Logger for debugging"""


class CorrelationBuilder(Builder):
    """
    A class to calculate the correlation between properties derived by or used in propnet
    using a suite of regression tools. Uses the Builder architecture for optional parallel
    processing of data.

    Notes:
        - Serialization of builder using ``as_dict()`` does not work with custom correlation
          functions, although interactive use does support them.

    Attributes:
        propnet_store (maggma.stores.Store): MongoDB collection containing quantity or material
            data to be queried for correlation calculation.
        correlation_store (maggma.stores.Store): MongoDB collection to which correlation calculation
            data will be written.
        from_quantity_db (bool): ``True`` if ``propnet_store`` follows the quantity-based schema created
            with ``propnet.dbtools.separation.SeparationBuilder``. ``False`` if ``propnet_store`` follows
            the material-based schema which is default output from ``propnet.dbtools.mp_builder.PropnetBuilder``.
        out_file (`str` or `None`): file name to output correlation data after builder is complete.
        sample_size (`int` or `None`): maximum number of randomly-sampled data points to include in
            correlation calculations. If ``None``, no limit is imposed.
    """
    _SCALAR_PROPS = [v.name for v in Registry("symbols").values()
                     if (v.category == 'property' and v.shape == 1)]
    """List of the names of all scalar properties contained in propnet."""
    
    def __init__(self, propnet_store,
                 correlation_store, out_file=None,
                 funcs='linlsq', props=None,
                 sample_size=None, from_quantity_db=True,
                 **kwargs):
        """
        Args:
            propnet_store (maggma.stores.Store): store instance pointing to propnet collection
                with read access
            correlation_store (maggma.stores.Store): store instance pointing to collection with write access
            out_file (str): optional, filename to output data in JSON format (useful if using a ``MemoryStore``
                for ``correlation_store``). Default: None (no file output)
            funcs (`str`, `callable`, `list` of `str` and/or `callable`): functions to use for correlation. Custom
                functions can be passed into this argument but are not JSON-serializable.
                Built-in functions can be specified by the following strings:

                - ``'linlsq'``: linear least-squares, reports R^2 (default)
                - ``'pearson'``: Pearson r-correlation, reports r
                - ``'spearman'``: Spearman rank correlation, reports r
                - ``'mic'``: maximal-information non-parametric exploration, reports maximal information coefficient
                - ``'ransac'``: random sample consensus (RANSAC) regression, reports score
                - ``'theilsen'``: Theil-Sen regression, reports score
                - ``'all'``: runs all correlation functions above

                Default: ``'linlsq'``
            props (`list` of `str`): optional, list of properties for which to calculate the correlation.
                Default: ``None`` (calculate for all possible pairs)
            sample_size (int): optional, limits correlation calculation data to a random sample of size
                `sample_size`. Default: ``None`` (no limit)
            from_quantity_db (bool): ``True`` means ``propnet_store`` follows the quantity-indexed database
                schema, ``False`` means the full, material-indexed database schema. Note: querying quantity-indexed
                databases is considerably faster than material-indexed.
                Default: ``True`` (quantity schema)
            **kwargs: arguments to the ``Builder`` superclass

        """

        self.propnet_store = propnet_store
        self.from_quantity_db = from_quantity_db
        self.correlation_store = correlation_store
        self.out_file = out_file

        self._correlation_funcs = self.get_correlation_funcs()

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

        self._props = props or self._SCALAR_PROPS

        if sample_size is not None and sample_size < 2:
            raise ValueError("Sample size must be greater than 1")
        self.sample_size = sample_size

        super(CorrelationBuilder, self).__init__(sources=[propnet_store],
                                                 targets=[correlation_store],
                                                 **kwargs)

    @property
    def total(self):
        """
        int: total number of calculations that will be performed during build
        """
        return len(self._props) ** 2 * len(self._funcs)

    @classmethod
    def get_correlation_funcs(cls):
        """
        Gets built-in correlation functions and their names.

        Returns:
            dict: dictionary of function handles keyed by name

        """
        return {f.replace('_cfunc_', ''): getattr(cls, f)
                for f in dir(cls)
                if re.match(r'^_cfunc_.+$', f) and callable(getattr(cls, f))}
    
    def get_items(self):
        """
        Accumulates data and generates data sets for pairs of properties coupled
        with correlation functions.

        Yields:
            dict: yields dictionaries of data with the following structure:

            - ``'x_data'`` (`list` of `float`): data for independent property (x-axis)
            - ``'x_name'`` (str): name of independent property
            - ``'y_data'`` (`list` of `float`): data for dependent property (y-axis)
            - ``'y_name'`` (str): name of dependent property
            - ``'func'`` (Tuple[str, callable]): name and function handle for correlation function
        """

        # combinations_with_replacement() produces all possible pairs of properties
        # without repeating, i.e. will give AB but not BA. Code below manually
        # produces "BA" so that we don't have to re-query the database.
        for prop_x, prop_y in combinations_with_replacement(self._props, 2):
            if self.from_quantity_db:
                data = self.get_data_from_quantity_db(self.propnet_store,
                                                      prop_x, prop_y,
                                                      sample_size=self.sample_size)
            else:
                data = self.get_data_from_full_db(prop_x, prop_y)

            yield from self._make_data_combinations(prop_x, prop_y, data)

    @staticmethod
    def get_data_from_quantity_db(store, *props, sample_size=None, include_id=False):
        """
        Collects scalar data from the quantity-only propnet database,
        aggregates it by material and property, and samples it if desired.

        Args:
            store (maggma.stores.Store): MongoDB store instance for quantity database
            *props (str): one or more property names as strings
            sample_size (int): if specified, limits the number of returned records
                to ``sample_size``, randomly selected. If total of records is less than
                ``sample_size``, only those records are returned. Default: ``None`` (all records)
            include_id (bool): ``True`` includes the ``_id`` field, which contains the material
                key for the record. Default: ``False`` (do not include the field)

        Returns:
            dict: dictionary of data keyed by property name

        """

        # This aggregation query collects the quantities, groups them by material
        # and averages the values for that material, then samples them (if specified)
        match_stage = {
            '$match': {
                '$or': [
                    {'symbol_type': prop} for prop in props
                ]}
        }
        group_stage = {'$group': {'_id': '$material_key'}}
        for prop in props:
            group_stage['$group'].update({
                prop: {
                    '$avg': {
                        '$cond': [
                            {"$eq": ['$symbol_type', prop]},
                            '$value',
                            None
                        ]
                    }
                }
            })
        pipeline = [match_stage, group_stage]

        if sample_size is not None:
            pipeline.append(
                {'$sample': {'size': sample_size}}
            )

        query = store.collection.aggregate(
            pipeline=pipeline,
            allowDiskUse=True
        )

        data = defaultdict(list)
        for m in query:
            if all(m[prop] is not None and np.isfinite(m[prop])
                   for prop in props):
                for prop in props:
                    data[prop].append(m[prop])
                if include_id:
                    data['_id'].append(m['_id'])

        return dict(data)

    def get_data_from_full_db(self, prop_x, prop_y):
        # TODO: Extract this and the quantity db equivalent as propnet db API class
        """
        Collects scalar data from full propnet database, aggregates it by property,
        and samples it if desired.

        Args:
            prop_x (str): name of property x
            prop_y (str): name of property y

        Returns:
            dict: dictionary of data keyed by property name

        """

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

        return data

    def _make_data_combinations(self, prop_x, prop_y, data):
        """
        Generates combinations of properties and desired correlation functions for evaluation.

        Args:
            prop_x (str): name of property x
            prop_y (str): name of property y
            data (dict): dictionary of data keyed by property name

        Yields:
            dict: a dictionary with the data for correlation with the following structure:

            - ``'x_data'`` (`list` of `float`): data for independent property (x-axis),
            - ``'x_name'`` (str): name of independent property,
            - ``'y_data'`` (`list` of `float`): data for dependent property (y-axis),
            - ``'y_name'`` (str): name of dependent property,
            - ``'func'`` (Tuple[str, callable]): name and function handle for correlation function

        """
        # So we get AB and BA without re-querying, but not two AA
        if prop_x == prop_y:
            prop_combos = ((prop_x, prop_x),)
        else:
            prop_combos = ((prop_x, prop_y), (prop_y, prop_x))
        for x, y in prop_combos:
            for name, func in self._funcs.items():
                data_dict = {'x_data': data.get(x, []),
                             'x_name': x,
                             'y_data': data.get(y, []),
                             'y_name': y,
                             'func': (name, func)}
                yield data_dict

    def process_item(self, item):
        """
        Runs correlation calculation on a pair of properties using the specified function.

        Args:
            item (dict): input provided by ``get_items()`` (see definition for structure)

        Returns:
            Tuple[str, str, float `or` Exception, str, int]: output of calculation with necessary
            information about calculation included. Ordering of elements in tuple is:

            - independent property (x-axis) name
            - dependent property (y-axis) name
            - correlation value or exception if an error occurs
            - correlation function name
            - number of data points used for correlation
            - length of shortest path between properties on propnet graph where x-axis property
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
        Gets maximal information coefficient for data set.

        Args:
            x (`list` of `float`): independent property (x-axis)
            y (`list` of `float`): dependent property (y-axis)

        Returns:
            float: maximal information coefficient

        """
        from minepy import MINE
        m = MINE()
        m.compute_score(x, y)
        return m.mic()

    @staticmethod
    def _cfunc_linlsq(x, y):
        """
        Gets R^2 value for linear least-squares fit of a data set.

        Args:
            x (`list` of `float`): independent property (x-axis)
            y (`list` of `float`): dependent property (y-axis)

        Returns:
            float: R^2 value

        """
        from scipy import stats
        fit = stats.linregress(x, y)
        return fit.rvalue ** 2

    @staticmethod
    def _cfunc_pearson(x, y):
        """
        Gets R value for Pearson fit of a data set.

        Args:
            x (`list` of `float`): independent property (x-axis)
            y (`list` of `float`): dependent property (y-axis)

        Returns:
            float: Pearson R value

        """
        from scipy import stats
        fit = stats.pearsonr(x, y)
        return fit[0]

    @staticmethod
    def _cfunc_spearman(x, y):
        """
        Gets R value for Spearman fit of a data set.

        Args:
            x (`list` of `float`): independent property (x-axis)
            y (`list` of `float`): dependent property (y-axis)

        Returns:
            float: Spearman R value

        """
        from scipy import stats
        fit = stats.spearmanr(x, y)
        return fit[0]

    @staticmethod
    def _cfunc_ransac(x, y):
        """
        Gets random sample consensus (RANSAC) regression score for data set.

        Args:
            x (`list` of `float`): independent property (x-axis)
            y (`list` of `float`): dependent property (y-axis)

        Returns:
            float: RANSAC score

        """
        from sklearn.linear_model import RANSACRegressor
        r = RANSACRegressor(random_state=21)
        x_coeff = np.array(x)[:, np.newaxis]
        r.fit(x_coeff, y)
        return r.score(x_coeff, y)

    @staticmethod
    def _cfunc_theilsen(x, y):
        """
        Gets Theil-Sen regression score for data set.

        Args:
            x (`list` of `float`): independent property (x-axis)
            y (`list` of `float`): dependent property (y-axis)

        Returns:
            float: Theil-Sen score

        """
        from sklearn.linear_model import TheilSenRegressor
        r = TheilSenRegressor(random_state=21)
        x_coeff = np.array(x)[:, np.newaxis]
        r.fit(x_coeff, y)
        return r.score(x_coeff, y)

    def update_targets(self, items):
        """
        Writes correlation data to Mongo store.

        Args:
            items (`list` of `tuple`): list of results output by ``process_item()``

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
            cursor (pymongo.cursor.Cursor): optional, cursor to close if not automatically closed.
                Default: ``None`` (no cursor to be closed)

        """

        props_to_index = ['property_x', 'property_y', 'correlation_func',
                          'correlation', 'shortest_path_length']
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
        Writes data from ``get_correlation_matrices()`` to a JSON file.

        Args:
            out_file (str): file path and name of JSON file to write
        """
        matrix = self.get_correlation_matrices()
        with open(out_file, 'w') as f:
            json.dump(matrix, f)

    def get_correlation_matrices(self, func_name=None):
        """
        Builds document containing the correlation matrix with relevant data regarding
        correlation algorithm and properties of the data set.

        Args:
            func_name (str): optional, name of the correlation functions to include in the document.
                Default: ``None`` (include all that were run by this builder)

        Returns:
            dict: dictionary containing correlation data with the following structure:

            - ``'properties'`` (`list` of `str`) - names of properties calculated in order of how
              they are indexed in the matrices
            - ``'n_points'`` (`list` of `list` of `int`) - list of lists (i.e. matrix) containing
              the number of data points evaluated during the fitting procedure
            - ``'correlation'`` (dict) - dictionary of matrices (list of list of floats) containing
              correlation results, keyed by correlation function name
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

        Returns:
            dict: representation of this builder as a JSON-serializable dictionary

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
