import numpy as np
from monty.json import MSONable
from monty.serialization import MontyDecoder
from datetime import datetime
import sys
from chronic import Timer

import networkx as nx

from abc import ABC, abstractmethod

from propnet import ureg
from pint import DimensionalityError, UnitRegistry
from propnet.core.symbols import Symbol
from propnet.core.provenance import ProvenanceElement

# noinspection PyUnresolvedReferences
import propnet.symbols
from propnet.core.registry import Registry

from propnet.core.exceptions import SymbolConstraintError
from typing import Union

import uuid
import copy
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class BaseQuantity(ABC, MSONable):
    """
    Base class for storing the value of a property.

    Subclasses of BaseQuantity allow for different kind of information to be stored and interpreted.

    Attributes:
        symbol_type: (Symbol or str) the type of information that is represented
            by the associated value.  If a string, assigns a symbol from
            the default symbols that has that string name
        value: (id) the value associated with this symbol. This can be any object.
        tags: (list<str>) tags associated with the quantity, typically
            related to its origin, e. g. "DFT" or "ML" or "experiment"
        provenance: (ProvenanceElement) provenance information of quantity origin
    """

    def __init__(self, symbol_type, value, tags=None,
                 provenance=None):
        """
        Parses inputs for constructing a BaseQuantity object.

        Args:
            symbol_type (Symbol or str): pointer to a Symbol
                object in Registry("symbols") or string giving the name
                of a Symbol object. Identifies the type of data
                stored in the quantity.
            value (id): value of the quantity.
            tags (list<str>): list of strings storing metadata from
                evaluation.
            provenance (ProvenanceElement): provenance associated with the
                object (e. g. inputs, model, see ProvenanceElement). If not specified,
                a default object will be created. All objects will receive
                the time created and the internal ID as fields 'source.date_created'
                and 'source.source_key', respectively, if the fields are not already
                written.
        """

        if not isinstance(symbol_type, Symbol):
            symbol_type = self.get_symbol_from_string(symbol_type)

        if provenance and not isinstance(provenance, ProvenanceElement):
            raise TypeError("Expected ProvenanceElement for provenance. "
                            "Instead received: {}".format(type(provenance)))

        self._value = value
        self._symbol_type = symbol_type
        self._tags = []
        if tags:
            if isinstance(tags, str):
                tags = [tags]
            self._tags.extend(tags)
        self._provenance = provenance
        self._internal_id = uuid.uuid4().hex

        if self._provenance is not None:
            if not isinstance(self._provenance.source, dict):
                self._provenance.source = {"source": self._provenance.source}

            if 'date_created' not in self._provenance.source.keys() or \
                    self._provenance.source['date_created'] in (None, ""):
                self._provenance.source['date_created'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if 'source_key' not in self._provenance.source.keys() or \
                    self._provenance.source['source_key'] in (None, ""):
                self._provenance.source['source_key'] = self._internal_id
        else:
            self._provenance = ProvenanceElement(source={"source": None,
                                                         "source_key": self._internal_id,
                                                         "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    @staticmethod
    def get_symbol_from_string(name):
        """
        Looks up Symbol from name in Registry("symbols") registry.

        Args:
            name: (str) the name of the Symbol object

        Returns: (Symbol) the Symbol object associated with the name

        """
        # Invoke default symbol if symbol is a string
        if not isinstance(name, str):
            raise TypeError("Expected str, encountered {}".format(type(name)))

        if name not in Registry("symbols").keys():
            raise ValueError("Symbol type {} not recognized".format(name))

        return Registry("symbols")[name]

    @property
    @abstractmethod
    def magnitude(self):
        """
        Returns the value of a quantity without any units.
        Should be implemented for numerical subclasses. Otherwise call self.value.

        Returns:
            (id): value without units (if numerical), otherwise just the value
        """
        pass

    @property
    def symbol(self):
        """
        Returns the Symbol object associated with the quantity.

        Returns:
            (Symbol): Symbol of the BaseQuantity
        """
        return self._symbol_type

    @property
    def tags(self):
        """
        Returns the list of tags.

        Returns:
            (list<str>): tags of the BaseQuantity
        """
        return self._tags

    @property
    def provenance(self):
        """
        Returns the object containing the provenance information for the quantity

        Returns:
            (ProvenanceElement): Provenance object for the quantity
        """
        return self._provenance

    @property
    def value(self):
        """
        Returns a copy of the value object stored in the quantity.

        Returns:
            (id): copy of value object stored in quantity
        """
        # This returns a deep copy of the object holding the value
        # in case it is a class instance and the user manipulates
        # the object. This is particularly problematic if a user
        # calls np.isclose(x.value, y.value) and x and/or y contain
        # pint Quantities. pint automatically converts the magnitudes
        # into ndarrays, even for scalars, which breaks the careful
        # type controlling we do for NumQuantity.
        # If this is problematic for large ndarrays or pymatgen objects,
        # for example, then we can revisit this decision to copy.
        return copy.deepcopy(self._value)

    @property
    @abstractmethod
    def units(self):
        """
        Returns the units of the quantity.
        Should be implemented for numerical subclasses. Otherwise return None.

        Returns:
            (pint.unit): units associated with the value
        """
        pass

    @property
    @abstractmethod
    def uncertainty(self):
        """
        Returns the pint object holding the uncertainty of a quantity.
        Should be implemented for numerical subclasses. Otherwise return None.

        Returns:
            (pint.Quantity): copy of uncertainty object stored in quantity
        """
        pass

    @abstractmethod
    def pretty_string(self, **kwargs):
        """
        Returns a string representing the value of the object in a pretty format.

        Returns:
            (str): text string representing the value of an object
        """
        pass

    def is_cyclic(self):
        """
        Algorithm for determining if there are any cycles in
        the provenance tree, i. e. a repeated quantity in a
        tree branch

        Returns:
            (bool) whether or not there is a cycle in the quantity
                provenance, i. e. repeated value in a tree branch
        """

        if self.provenance and self.provenance.model:
            return self.provenance.model_in_provenance_tree(self.provenance.model) or \
                self.provenance.symbol_in_provenance_tree(self.symbol)

        return False

    def get_provenance_graph(self, start=None, filter_long_labels=True):
        """
        Gets an nxgraph object corresponding to the provenance graph

        Args:
            start (nxgraph): starting graph to build from
            filter_long_labels (bool): true truncates long labels to just the symbol name

        Returns:
            (nxgraph): graph representation of provenance
        """
        graph = start or nx.MultiDiGraph()
        label = "{}: {}".format(self.symbol.name, self.pretty_string())
        if filter_long_labels and len(label) > 30:
            label = "{}".format(self.symbol.name)
        graph.add_node(
            self, fillcolor="#43A1F8", fontcolor='white', label=label)
        model = getattr(self.provenance, 'model', None)
        source = getattr(self.provenance, 'source', None)
        if model is not None:
            model = "Model: {}".format(model)
            graph.add_node(model, label=model, fillcolor='orange',
                           fontcolor='white', shape='rectangle')
            graph.add_edge(model, self)
            for model_input in self.provenance.inputs:
                graph = model_input.get_provenance_graph(start=graph)
                graph.add_edge(model_input, model)
        elif source is not None:
            source = "Source: {}".format(source['source'])
            graph.add_edge(source, self)

        return graph

    def draw_provenance_graph(self, filename, prog='dot', **kwargs):
        """
        Outputs the provenance graph for this quantity to a file.

        Args:
            filename: (str) filename for output
            prog: (str) pygraphviz layout method for drawing the graph
            **kwargs: args to pygraphviz.AGraph.draw() method
        """
        nx_graph = self.get_provenance_graph()
        a_graph = nx.nx_agraph.to_agraph(nx_graph)
        a_graph.node_attr['style'] = 'filled'
        a_graph.draw(filename, prog=prog, **kwargs)

    def as_dict(self):
        """
        Serializes object as a dictionary. Object can be reconstructed with from_dict().

        Returns:
            (dict): representation of object as a dictionary
        """
        symbol = self._symbol_type
        if symbol.name in Registry("symbols").keys() and symbol == Registry("symbols")[symbol.name]:
            symbol = self._symbol_type.name
        else:
            symbol = symbol.as_dict()

        return {"symbol_type": symbol,
                "provenance": self.provenance.as_dict() if self.provenance else None,
                "tags": self.tags,
                "internal_id": self._internal_id}

    @classmethod
    def from_dict(cls, d):
        decoded = {k: MontyDecoder().process_decoded(v) for k, v in d.items() if not k.startswith('@')}
        internal_id = decoded.pop('internal_id')
        value = decoded.pop('value')
        symbol_type = decoded.pop('symbol_type')
        q = cls(symbol_type, value, **decoded)
        q._internal_id = internal_id
        return q

    @abstractmethod
    def contains_nan_value(self):
        """
        Determines if value contains a NaN (not a number) value.
        Should be implemented for numerical subclasses. Otherwise return False.

        Returns:
            (bool): True if value contains at least one NaN value.
        """
        pass

    @abstractmethod
    def contains_complex_type(self):
        """
        Determines if value contains one or more complex-type values based on variable type.
        Should be implemented for numerical subclasses. Otherwise return False.

        Returns:
            (bool): True if value contains at least one complex-type value.
        """
        pass

    @abstractmethod
    def contains_imaginary_value(self):
        """
        Determines if value has a non-zero imaginary component. Differs from
        contains_complex_type() in that it checks the imaginary component's value.
        If zero or very small, returns True.

        Should be implemented for numerical subclasses. Otherwise return False.

        Returns:
            (bool): True if value contains at least one value with a non-zero imaginary component.
        """
        pass

    @abstractmethod
    def has_eq_value_to(self, rhs):
        """
        Determines if the current quantity's value is equal to that of another quantity.
        This ignores provenance of the quantity and compares the values only.

        Args:
            rhs: (BaseQuantity) the quantity to which the current object will be compared

        Returns: (bool): True if the values are found to be equal (or equivalent)

        """
        pass

    def __hash__(self):
        """
        Hash function for this class.

        Note: the hash function for this class does not hash the value,
            so it cannot alone determine equality.

        Returns: (int) hash value

        """
        hash_value = hash(self.symbol.name) ^ hash(self.provenance)
        if self.tags:
            # Sorting to ensure it is deterministic
            sorted_tags = self.tags.copy()
            sorted_tags.sort()
            for tag in sorted_tags:
                hash_value = hash_value ^ hash(tag)
        return hash_value

    def __str__(self):
        return "<{}, {}, {}>".format(self.symbol.name, self.value, self.tags)

    def __repr__(self):
        return self.__str__()

    def __bool__(self):
        return bool(self.value)

    def __eq__(self, other):
        """
        Determines equality of common components of BaseQuantity-derived objects.

        Note: Does not check for equivalence of value. Derived classes should
            override this method to determine equivalence of values.

        Note: __eq__() does not provide comparisons to other types, but does support
            implied comparisons by returning NotImplemented for other types.

        Args:
            other: (BaseQuantity-derived type) object for value comparison

        Returns: (bool) True if the symbol, tags, and provenance are equal.

        """
        return self.symbol == other.symbol and \
               self.tags == other.tags and \
               self.provenance == other.provenance


class NumQuantity(BaseQuantity):
    """
    Class extending BaseQuantity for storing numerical values, scalar and non-scalar.

    Allowed scalar types: int, float, complex, np.integer, np.floating, np.complexfloating
    Allowed array types: list, np.array

    Note: Array types must contain only allowed scalar types. Scalars with numpy types will
    be converted to python-native types.

    Types shown below are how the objects are stored. See __init__() for initialization.

    Attributes:
        symbol_type: (Symbol) the type of information that is represented
            by the associated value
        value: (pint.Quantity) the value of the property wrapped in a pint quantity
            for unit handling
        units: (pint.unit) units of the object
        tags: (list<str>) tags associated with the quantity, typically
            related to its origin, e. g. "DFT" or "ML" or "experiment"
        provenance: (ProvenanceElement) provenance associated with the
                object. See BaseQuantity.__init__() for more info.
        uncertainty: (pint.Quantity) uncertainty associated with the value stored in the same units
    """

    # Allowed types
    _ACCEPTABLE_SCALAR_TYPES = (int, float, complex)
    _ACCEPTABLE_ARRAY_TYPES = (list, np.ndarray)
    _ACCEPTABLE_DTYPES = (np.integer, np.floating, np.complexfloating)
    _ACCEPTABLE_TYPES = _ACCEPTABLE_ARRAY_TYPES + _ACCEPTABLE_SCALAR_TYPES + _ACCEPTABLE_DTYPES + (ureg.Quantity,)
    # This must be checked for explicitly because bool is a subtype of int
    # and isinstance(True/False, int) returns true
    _UNACCEPTABLE_TYPES = (bool,)

    def __init__(self, symbol_type, value, units=None, tags=None,
                 provenance=None, uncertainty=None):
        """
        Instantiates an instance of the NumQuantity class.

        Args:
            symbol_type: (Symbol or str) the type of information that is represented
                by the associated value.  If a string, assigns a symbol from
                the default symbols that has that string name
            value: (int, float, complex, np.integer, np.floating, np.complexfloating,
                list, np.ndarray, pint.Quantity) the value of the property
            units: (str, tuple, list) desired units of the quantity. If value is a
                pint.Quantity, the value will be converted to these units. Input can
                be any acceptable unit format for pint.Quantity.
            tags: (list<str>) tags associated with the quantity, typically
                related to its origin, e. g. "DFT" or "ML" or "experiment"
            provenance: (ProvenanceElement) provenance associated with the
                object. See BaseQuantity.__init__() for more info.
            uncertainty: (int, float, complex, np.integer, np.floating, np.complexfloating,
                list, np.ndarray, pint.Quantity, tuple, NumQuantity) uncertainty
                associated with the value stored in the same units. pint.Quantity,
                tuple, and NumQuantity types will be converted to the units
                specified in 'units'. Other types will be assumed to be in the
                specified units.

        """
        # TODO: Test value on the shape dictated by symbol
        with Timer('get_symbol'):
            if isinstance(symbol_type, str):
                symbol_type = BaseQuantity.get_symbol_from_string(symbol_type)

        # Set default units if not supplied
        with Timer('get_units'):
            if not units:
                logger.warning("No units supplied, assuming default units from symbol.")

            units = units or symbol_type.units

        with Timer('coerce_value'):
            if isinstance(value, self._ACCEPTABLE_DTYPES):
                value_in = ureg.Quantity(value.item(), units)
            elif isinstance(value, ureg.Quantity):
                value_in = value.to(units)
            elif self.is_acceptable_type(value):
                value_in = ureg.Quantity(value, units)
            else:
                raise TypeError('Invalid type passed to constructor for value:'
                                ' {}'.format(type(value)))

        with Timer('run_super_init'):
            super(NumQuantity, self).__init__(symbol_type, value_in,
                                              tags=tags, provenance=provenance)

        with Timer('coerce_uncertainty'):
            if uncertainty is not None:
                if isinstance(uncertainty, self._ACCEPTABLE_DTYPES):
                    self._uncertainty = ureg.Quantity(uncertainty.item(), units)
                elif isinstance(uncertainty, ureg.Quantity):
                    self._uncertainty = uncertainty.to(units)
                elif isinstance(uncertainty, NumQuantity):
                    self._uncertainty = uncertainty._value.to(units)
                elif isinstance(uncertainty, tuple):
                    self._uncertainty = ureg.Quantity.from_tuple(uncertainty).to(units)
                elif self.is_acceptable_type(uncertainty):
                    self._uncertainty = ureg.Quantity(uncertainty, units)
                else:
                    raise TypeError('Invalid type passed to constructor for uncertainty:'
                                    ' {}'.format(type(uncertainty)))
            else:
                self._uncertainty = None

        # TODO: Symbol-level constraints are hacked together atm,
        #       constraints as a whole need to be refactored and
        #       put into a separate module. They also are only
        #       available for numerical symbols because it uses
        #       sympy to evaluate the constraints. Would be better
        #       to make some class for symbol and/or model constraints

        with Timer('evaluate_constraint'):
            symbol_constraint = symbol_type.constraint
            if symbol_constraint is not None:
                if not symbol_constraint(**{symbol_type.name: self.magnitude}):
                    raise SymbolConstraintError(
                        "NumQuantity with {} value does not satisfy {}".format(
                            value, symbol_constraint))

    @staticmethod
    def _is_acceptable_dtype(this_dtype):
        """
        This function checks a dtype against the allowed dtypes for this class.

        Args:
            this_dtype: (numpy.dtype) the dtype to check

        Returns: True if this_dtype is a sub-dtype of the acceptable dtypes.

        """
        return any([np.issubdtype(this_dtype, dt) for dt in NumQuantity._ACCEPTABLE_DTYPES])

    def to(self, units):
        """
        Method to convert quantities between units, a la pint

        Args:
            units: (tuple or str) units to convert quantity to

        Returns:

        """
        # Calling deepcopy() instead of ctor preserves internal_id
        # while returning a new object (as is desired?)
        q = copy.deepcopy(self)
        q._value = q._value.to(units)
        if q._uncertainty is not None:
            q._uncertainty = q._uncertainty.to(units)
        return q

    @classmethod
    def from_weighted_mean(cls, quantities):
        """
        Function to invoke weighted mean quantity from other
        quantities

        Args:
            quantities ([NumQuantity]): list of quantities of the same type

        Returns: (NumQuantity) a quantity containing the weighted mean and
            standard deviation.
        """

        if not all(isinstance(q, cls) for q in quantities):
            raise ValueError("Weighted mean cannot be applied to non-NumQuantity objects")

        input_symbol = quantities[0].symbol
        if not all(input_symbol == q.symbol for q in quantities):
            raise ValueError("Can only calculate a weighted mean if "
                             "all quantities refer to the same symbol.")

        # TODO: an actual weighted mean; just a simple mean at present
        # TODO: support propagation of uncertainties (this will only work
        # once at present)

        # # TODO: test this with units, not magnitudes ... remember units
        # # may not be canonical units(?)
        # if isinstance(quantities[0].value, list):
        #     # hack to get arrays working for now
        #     vals = [q.value for q in quantities]
        # else:
        #     vals = [q.value.magnitude for q in quantities]
        vals = [q.value for q in quantities]

        # Explicit formulas for mean / standard dev for pint support
        new_value = sum(vals) / len(vals)
        std_dev = (sum([(v - new_value) ** 2 for v in vals]) / len(vals)) ** (1 / 2)

        # Accumulate provenance and tags for new quantities
        new_tags = set()
        new_provenance = ProvenanceElement(model='aggregation', inputs=[])
        for quantity in quantities:
            if quantity.tags:
                for tag in quantity.tags:
                    new_tags.add(tag)
            new_provenance.inputs.append(quantity)

        return cls(symbol_type=input_symbol, value=new_value,
                   tags=list(new_tags), provenance=new_provenance,
                   uncertainty=std_dev)

    @property
    def magnitude(self):
        """
        Returns the value of a quantity without any units.

        Returns:
            (int, float, complex, np.ndarray): value without units
        """
        return self._value.magnitude

    @property
    def units(self):
        """
        Returns the units of the quantity.

        Returns:
            (pint.unit): units associated with the value
        """
        return self._value.units

    @property
    def uncertainty(self):
        """
        Returns the pint object holding the uncertainty of a quantity.

        Returns:
            (pint.Quantity): copy of uncertainty object stored in quantity
        """
        # See note on BaseQuantity.value about why this is a deep copy
        return copy.deepcopy(self._uncertainty)

    @staticmethod
    def is_acceptable_type(to_check):
        """
        Checks object to ensure it contains only numerical types, including numpy types.
        Works with nested lists.

        Args:
            to_check: (list) list of data to be checked

        Returns: (bool): true if all data contained in the list is numerical (float, int, complex)

        """
        def recursive_list_type_check(l):
            nested_lists = [v for v in l if isinstance(v, list)]
            np_arrays = [v for v in l if isinstance(v, np.ndarray)]
            ureg_quantities = [v for v in l if isinstance(v, ureg.Quantity)]
            regular_data = [v for v in l if not isinstance(v, (list, np.ndarray))]

            regular_data_is_type = all([isinstance(v, NumQuantity._ACCEPTABLE_TYPES) and not
                                        isinstance(v, NumQuantity._UNACCEPTABLE_TYPES)
                                        for v in regular_data])
            # If nested_lists is empty, all() returns true automatically
            nested_lists_is_type = all(recursive_list_type_check(v) for v in nested_lists)
            np_arrays_is_type = all(NumQuantity._is_acceptable_dtype(v.dtype)
                                    for v in np_arrays)

            ureg_quantities_is_type = all(recursive_list_type_check([v.magnitude])
                                          for v in ureg_quantities)

            return regular_data_is_type and nested_lists_is_type \
                   and np_arrays_is_type and ureg_quantities_is_type

        return recursive_list_type_check([to_check])

    def pretty_string(self, **kwargs):
        """
        Returns a string representing the value of the object in a pretty format with units.
        Note: units are omitted for non-scalar properties.

        Keyword Args:
            sigfigs: (int) how many significant figures to include. default: 4
        Returns:
            (str): text string representing the value of an object
        """

        # TODO: maybe support a rounding kwarg?
        if 'sigfigs' in kwargs.keys():
            sigfigs = kwargs['sigfigs']
        else:
            sigfigs = 4
        if isinstance(self.magnitude, self._ACCEPTABLE_SCALAR_TYPES):
            out = "{1:.{0}g}".format(sigfigs, self.magnitude)
            if self.uncertainty:
                out += "\u00B1{0:.4g}".format(self.uncertainty.magnitude)
        else:
            out = "{}".format(self.magnitude)

        if self.units and str(self.units) != 'dimensionless':
            # The format str is specific to pint units. ~ invokes abbreviations, P is "pretty" format
            out += " {:~P}".format(self.units)

        return out

    def contains_nan_value(self):
        """
        Determines if the value of the object contains a NaN value if the
        object holds numerical data.

        Returns:
             (bool) true if the quantity is numerical and contains one
             or more NaN values. false if the quantity is numerical and
             does not contain any NaN values OR if the quantity does not
             store numerical information
        """

        return np.any(np.isnan(self.magnitude))

    def contains_complex_type(self):
        """
        Determines if the type of the variable holding the object's magnitude is complex, if the
        object holds numerical data.

        Returns:
             (bool) true if the quantity is numerical and holds a complex scalar or array type as its value.
             false if the quantity is numerical and holds only real values OR if the quantity does not
             store numerical information
        """

        return self.is_complex_type(self.magnitude)

    @staticmethod
    def is_complex_type(value):
        """
        Determines if the type of the argument is complex. If the argument is non-scalar, it determines
        if the ndarray type contains complex data types.

        Returns:
             (bool) true if the argument holds a complex scalar or np.array.

        """
        if isinstance(value, np.ndarray):
            return np.issubdtype(value.dtype, np.complexfloating)
        elif isinstance(value, BaseQuantity):
            return value.contains_complex_type()
        elif isinstance(value, ureg.Quantity):
            return NumQuantity.is_complex_type(value.magnitude)

        return isinstance(value, complex)

    def contains_imaginary_value(self):
        """
        Determines if the value of the object contains a non-zero imaginary
        value if the object holds numerical data.

        Note this function returns false if the values are of complex type,
        but the imaginary portions are (approximately) zero. To assess the
        type as complex, use is_complex_type().

        Returns:
             (bool) true if the quantity is numerical and contains one
             or more non-zero imaginary values. false if the quantity is
             numerical and all imaginary values are zero OR if the quantity does not
             store numerical information.
        """
        if self.contains_complex_type():
            # Calling as static methods allows for evaluation of both scalars and arrays
            return not np.all(np.isclose(np.imag(self.magnitude), 0))

        return False

    def as_dict(self):
        """
        Serializes object as a dictionary. Object can be reconstructed with from_dict().

        Returns:
            (dict): representation of object as a dictionary
        """
        d = super(NumQuantity, self).as_dict()

        d.update({"@module": self.__class__.__module__,
                  "@class": self.__class__.__name__,
                  "value": self.magnitude,
                  "units": self.units.format_babel(),
                  "uncertainty": self.uncertainty.to_tuple() if self.uncertainty else None})

        return d

    def __eq__(self, other):
        """
        Determines if another NumQuantity object is equivalent to this object.

        Equivalence is defined as having the same symbol name, tags, provenance, and
        equal (within tolerance) value and uncertainty in the default units of the symbol.

        Note: __eq__() does not provide comparisons to other types, but does support
            implied comparisons by returning NotImplemented for other types.

        Args:
            other: (NumQuantity) the object to compare to

        Returns: (bool) True if the objects are equivalent

        """
        # Use has_eq_value_to() to compare only values.
        if not isinstance(other, NumQuantity):
            return NotImplemented
        if not self.uncertainty and not other.uncertainty:
            uncertainty_is_close = True
        elif self.uncertainty and other.uncertainty:
            uncertainty_is_close = self.values_close_in_units(self.uncertainty,
                                                              other.uncertainty,
                                                              units_for_comparison=self.uncertainty.units)
        else:
            return False

        value_is_close = self.values_close_in_units(self.value, other.value,
                                                    units_for_comparison=self.symbol.units)

        return \
            super().__eq__(other) and \
            uncertainty_is_close and \
            value_is_close

    def has_eq_value_to(self, rhs):
        """
        Determines if the current quantity's value is equivalent to that of another quantity.
        This ignores provenance of the quantity and compares the values only.

        Equivalence is defined as having the same numerical value in the units defined by the
        quantities' symbol, within an absolute tolerance of 1e-8 and relative tolerance of 1e-5.

        Args:
            rhs: (NumQuantity) the quantity to which the current object will be compared

        Returns: (bool): True if the values are found to be equivalent

        """
        if not isinstance(rhs, type(self)):
            raise TypeError("This method requires two {} objects".format(type(self).__name__))
        return self.values_close_in_units(self.value, rhs.value,
                                          units_for_comparison=self.symbol.units)

    @staticmethod
    def values_close_in_units(lhs, rhs, units_for_comparison=None):
        """
        Compares two pint quantities in a given unit. The purpose is to
        ensure dimensional, small quantities (e.g. femtoseconds) don't
        get discounted as small, close-to-zero quantities.

        If units are not specified explicitly, they are selected using the
        following criteria, in order of precedence:
        1. If one quantity has a value of exactly 0, the units of that quantity
            are used for comparison.
        2. The units of both quantities are rescaled such that the magnitude
            of each quantity is between 1 and 1000, or where the unit is at the
            smallest (or largest) possible unit defined by pint. The smaller of
            the two units is then used to compare the values (i.e. gram would be
            selected over kilogram).

        Note: dimensionless quantities will NOT be scaled and will be treated
            as bare numbers. This means dimensionless values that are small,
            but different will be treated as equal if abs(a-b) <= 1e-8, e.g.
            1e-8 and 2e-8 will yield True, as will 1e-8 and 1e-20.

        Args:
            lhs: (pint.Quantity) quantity object to compare
            rhs: (pint.Quantity) quantity object to compare
            units_for_comparison: (str, pint.Units, tuple) units that the
                quantities will be compared in. Input can be any acceptable
                format for Quantity.to()

        Returns: (bool) True if the values are equal within an absolute tolerance
            of 1e-8 and a relative tolerance of 1e-5. False if not equal within
            the tolerance bounds, or the dimensionality of the units are not equal.

        """
        if not (isinstance(lhs, ureg.Quantity) and isinstance(rhs, ureg.Quantity)):
            raise TypeError("This method requires two pint Quantity objects. "
                            "Received:\n{} == {}".format(type(lhs), type(rhs)))

        if lhs.units.dimensionality != rhs.units.dimensionality:
            return False

        if not units_for_comparison:
            if not isinstance(lhs.magnitude, np.ndarray):
                if lhs.magnitude == 0 and rhs.magnitude == 0:
                    return True
                elif lhs.magnitude == 0:
                    # Compare using the units of whatever the zero value is
                    units_for_comparison = lhs.units
                elif rhs.magnitude == 0:
                    units_for_comparison = rhs.units
                else:
                    # Select smallest unit that brings values close to 1
                    # Add a +1 buffer so that instead of 999.99999... micrograms
                    # we get 1 milligram instead.
                    lhs_compact = lhs.to_compact()
                    lhs_compact_units = (lhs_compact + 1 * lhs_compact.units).to_compact().units
                    rhs_compact = rhs.to_compact()
                    rhs_compact_units = (rhs_compact + 1 * rhs_compact.units).to_compact().units
                    if 1 * lhs_compact_units < 1 * rhs_compact_units:
                        units_for_comparison = lhs_compact_units
                    else:
                        units_for_comparison = rhs_compact_units
            else:
                try:
                    if 1 * lhs.units < 1 * rhs.units:
                        units_for_comparison = lhs.units
                    else:
                        units_for_comparison = rhs.units
                except DimensionalityError:
                    return False
        try:
            lhs_convert = lhs.to(units_for_comparison)
            rhs_convert = rhs.to(units_for_comparison)
        except DimensionalityError:
            return False
        return np.allclose(lhs_convert, rhs_convert)

    def __hash__(self):
        """
        Hash function for this class.

        Note: the hash function for this class does not hash the value,
            so it cannot alone determine equality.

        Returns: (int) hash value

        """
        return super().__hash__()

    def __getstate__(self):
        d = self.__dict__.copy()
        d['_value'] = d['_value'].to_tuple()
        if self._uncertainty is not None:
            d['_uncertainty'] = d['_uncertainty'].to_tuple()
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._value = ureg.Quantity.from_tuple(self._value)
        if self._uncertainty is not None:
            self._uncertainty = ureg.Quantity.from_tuple(self._uncertainty)


class ObjQuantity(BaseQuantity):
    """
    Class extending BaseQuantity for storing any value type that does not require units.

    Types shown below are how the objects are stored. See __init__() for initialization.

    Attributes:
        symbol_type: (Symbol) the type of information that is represented
            by the associated value
        value: (id) the value of the property
        tags: (list<str>) tags associated with the quantity, typically
            related to its origin, e. g. "DFT" or "ML" or "experiment"
        provenance: (ProvenanceElement) provenance associated with the
                object. See BaseQuantity.__init__() for more info.
    """

    def __init__(self, symbol_type, value, tags=None,
                 provenance=None):
        """
        Instantiates an instance of the ObjQuantity class.

        Args:
            symbol_type: (Symbol or str) the type of information that is represented
                by the associated value.  If a string, assigns a symbol from
                the default symbols that has that string name
            value: (id) the value of the property, can be any type except None.
                Ideally, numerical values should be stored in NumQuantity objects,
                because ObjQuantity does not support units.
            tags: (list<str>) tags associated with the quantity, typically
                related to its origin, e. g. "DFT" or "ML" or "experiment"
            provenance: (ProvenanceElement) provenance associated with the
                object. See BaseQuantity.__init__() for more info.
        """
        if value is None:
            raise ValueError("ObjQuantity must hold a non-NoneType object for its value.")

        if isinstance(symbol_type, str):
            symbol_type = super().get_symbol_from_string(symbol_type)

        if not symbol_type.is_correct_object_type(value):
            old_type = type(value)
            target_module = symbol_type.object_module
            target_class = symbol_type.object_class
            if target_module in sys.modules and \
                    hasattr(sys.modules[target_module], target_class):
                try:
                    cls_ = getattr(sys.modules[target_module], target_class)
                    value = cls_(value)
                except (TypeError, ValueError):
                    raise TypeError("Mismatch in type of value ({}) and type specified "
                                    "by '{}' object symbol ({}).\nTypecasting failed."
                                    "".format(old_type.__name__,
                                              symbol_type.name,
                                              symbol_type.object_class))
            else:
                # Do not try to import the module for security reasons.
                # We don't want malicious modules to be automatically imported.
                raise NameError("Mismatch in type of value ({}) and type specified "
                                "by '{}' object symbol ({}).\nCannot typecast because "
                                "'{}' is not imported or does not exist."
                                "".format(old_type.__name__,
                                          symbol_type.name,
                                          symbol_type.object_class,
                                          symbol_type.object_type))

            logger.warning("WARNING: Mismatch in type of value ({}) "
                           "and type specified by '{}' object symbol ({}). "
                           "Value cast as '{}'.".format(old_type.__name__,
                                                        symbol_type.name,
                                                        symbol_type.object_class,
                                                        symbol_type.object_type))

        super(ObjQuantity, self).__init__(symbol_type, value, tags=tags, provenance=provenance)

    @property
    def magnitude(self):
        """
        Returns the value of the quantity. Same as self.value.

        Returns:
            (id): value contained by quantity
        """
        return self._value

    @property
    def units(self):
        """
        Returns None because this class does not support units.

        Returns:
            None
        """
        return None

    @property
    def uncertainty(self):
        """
        Returns None because this class does not support uncertainty.

        Returns:
            None
        """
        return None

    def pretty_string(self, **kwargs):
        """
        Returns a string representing the value of the object in a pretty format.

        Returns:
            (str): text string representing the value of an object
        """
        return "{}".format(self.value)

    # TODO: Determine whether it's necessary to define these for ObjQuantity
    # we could just assess this if models return NumQuantity
    def contains_nan_value(self):
        """
        Returns False because this class does not support numerical types.

        Returns:
            (bool): False
        """
        return False

    def contains_complex_type(self):
        """
        Returns False because this class does not support numerical types.

        Returns:
            (bool): False
        """
        return False

    def contains_imaginary_value(self):
        """
        Returns False because this class does not support numerical types.

        Returns:
            (bool): False
        """
        return False

    def has_eq_value_to(self, rhs):
        """
        Determines if the value of another ObjQuantity is equivalent to the current.

        Equivalence is defined by the default __eq__() method for the object held in value.

        Args:
            rhs: (ObjQuantity) object for value comparison

        Returns: (bool) True if the values are equal.

        """
        if not isinstance(rhs, type(self)):
            raise TypeError("This method requires two {} objects".format(type(self).__name__))
        return self.value == rhs.value

    def as_dict(self):
        """
        Serializes object as a dictionary. Object can be reconstructed with from_dict().

        Note: If value is not JSON serializable, this object will not be JSON serializable.

        Returns:
            (dict): representation of object as a dictionary
        """
        d = super().as_dict()

        d.update({"@module": self.__class__.__module__,
                  "@class": self.__class__.__name__,
                  "value": self.value})

        return d

    def __eq__(self, other):
        """
        Determines if the value of another ObjQuantity is equivalent to the current.

        Equivalence is defined by equivalence of symbol name, tags, provenance, and
            value as indicated by the default __eq__() method for the object held
            in value.

        Note: __eq__() does not provide comparisons to other types, but does support
            implied comparisons by returning NotImplemented for other types.

        Args:
            other: (ObjQuantity) object for value comparison

        Returns: (bool) True if the objects are equal.

        """
        # Use has_eq_value_to() to compare only values.
        if not isinstance(other, ObjQuantity):
            return False

        return super().__eq__(other) and self.value == other.value

    def __hash__(self):
        """
        Hash function for this class.

        Note: the hash function for this class does not hash the value,
            so it cannot alone determine equality.

        Returns: (int) hash value

        """
        return super().__hash__()


class QuantityFactory(object):
    """
    Helper class to construct BaseQuantity-derived objects using factory methods.

    Use create_quantity() to generate objects on the fly depending on the value type.
    """
    @staticmethod
    def create_quantity(symbol_type, value, units=None, tags=None,
                        provenance=None, uncertainty=None):
        """
        Factory method for BaseQuantity class, provides more intuitive call
        to create new BaseQuantity child objects and provide backwards
        compatibility. If BaseQuantity is passed in as value, a new object
        will be created. Use BaseQuantity.to_quantity() to return the same object.

        Args:
            symbol_type: (str or Symbol) represents the type of data being stored
            value: (id) data to be stored
            units: (str, ureg.Unit) units of the data being stored. Must be None
                for non-numerical values
            tags: (list<str>) list of strings storing metadata from
                    Quantity evaluation.
            provenance: (ProvenanceElement) provenance associated with the
                    object (e. g. inputs, model, see ProvenanceElement)
            uncertainty: (id) uncertainty of the specified value

        Returns: (NumQuantity or ObjQuantity) constructed object of appropriate
            type based on value's type

        """
        if value is None:
            raise ValueError("Cannot initialize a BaseQuantity object with a value of None.")

        if isinstance(value, BaseQuantity):
            units = units or value.units
            tags = tags or value.tags
            provenance = provenance or value.provenance
            uncertainty = uncertainty or value.uncertainty
            value = value.value

        # TODO: This sort of thing probably indicates Symbol objects need to be split
        #       into different categories, like Quantity has been

        if not isinstance(symbol_type, Symbol):
            symbol_type = BaseQuantity.get_symbol_from_string(symbol_type)

        symbol_is_object = symbol_type.category == 'object'

        if not symbol_is_object:
            if NumQuantity.is_acceptable_type(value):
                return NumQuantity(symbol_type, value,
                                   units=units, tags=tags,
                                   provenance=provenance,
                                   uncertainty=uncertainty)
            else:
                raise TypeError("Cannot initialize a {}-type symbol with a non-numerical"
                                " value type.".format(symbol_type.category))

        if units is not None:
            logger.warning("Cannot assign units to object-type symbol '{}'. "
                           "Ignoring units.".format(symbol_type.name))

        if uncertainty is not None:
            logger.warning("Cannot assign uncertainty to object-type symbol '{}'. "
                           "Ignoring uncertainty.".format(symbol_type.name))

        return ObjQuantity(symbol_type, value,
                           tags=tags,
                           provenance=provenance)

    @staticmethod
    def from_default(symbol):
        """
        Method to invoke a default quantity from a symbol name

        Args:
            symbol (Symbol or str): symbol or string corresponding to
                the symbol name

        Returns:
            BaseQuantity corresponding to default quantity from default
        """
        val = Registry("symbol_values").get(symbol)
        if val is None:
            raise ValueError("No default value for {}".format(symbol))
        prov = ProvenanceElement(model='default')
        return QuantityFactory.create_quantity(symbol, val, provenance=prov)

    @staticmethod
    def to_quantity(symbol: Union[str, Symbol],
                    to_coerce: Union[float, np.ndarray, ureg.Quantity, "BaseQuantity"],
                    **kwargs) -> "BaseQuantity":
        """
        Converts the argument into a BaseQuantity-derived object. If input is:
        - BaseQuantity-derived object -> immediately returned without modification (same object, not copied)
        - Any other python object -> passed to create_quantity() with keyword arguments
            to create a new BaseQuantity-derived object

        Args:
            symbol: a string or Symbol object representing the type of data stored
            to_coerce: item to be converted into a BaseQuantity-derived object
            kwargs: keyword arguments to create new object if to_coerce is not a BaseQuantity-derived object
        Returns:
            (BaseQuantity) item as a BaseQuantity-derived object
        """
        # If a quantity is passed in, return the quantity.
        if isinstance(to_coerce, BaseQuantity):
            return to_coerce

        # Else
        # Return the correct BaseQuantity - warns if units are assumed.
        return QuantityFactory.create_quantity(symbol, to_coerce, **kwargs)

    @staticmethod
    def from_dict(d):
        """
        Method to construct BaseQuantity-derived objects from dictionaries.

        Args:
            d: (dict) input dictionary

        Returns: (NumQuantity, ObjQuantity) new object defined from dictionary

        """
        if d['@class'] == 'NumQuantity':
            return NumQuantity.from_dict(d)
        elif d['@class'] == 'ObjQuantity':
            return ObjQuantity.from_dict(d)
        else:
            raise ValueError("Cannot build non-BaseQuantity objects!")


