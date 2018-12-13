import numpy as np
from monty.json import MSONable
from datetime import datetime

import networkx as nx

from abc import ABC, abstractmethod

from propnet import ureg
from propnet.core.symbols import Symbol
from propnet.core.provenance import ProvenanceElement
from propnet.symbols import DEFAULT_SYMBOLS, DEFAULT_SYMBOL_VALUES

from propnet.core.exceptions import SymbolConstraintError
from typing import Union

import uuid
import copy
import logging

logger = logging.getLogger(__name__)

class BaseQuantity(ABC):
    """
    Class storing the value of a property.

    Constructed by the user to assign values to abstract Symbol types.
    Represents the fact that a given Quantity has a given value. They
    are added to the PropertyNetwork graph in the context of Material
    objects that store collections of Quantity objects representing
    that a given material has those properties.

    Attributes:
        symbol_type: (Symbol or str) the type of information that is represented
            by the associated value.  If a string, assigns a symbol from
            the default symbols that has that string name
        _value: (id) the value associated with this symbol.  Note that
            this should be either a pint quantity or an object.
        tags: (list<str>) tags associated with the quantity, typically
            related to its provenance, e. g. "DFT" or "ML"
    """

    def __init__(self, symbol_type, value, tags=None,
                 provenance=None):
        """
        Parses inputs for constructing a Property object.

        Args:
            symbol_type (Symbol or str): pointer to an existing Symbol
                object in default_symbols or string giving the name
                of a SymbolType object identifies the type of data
                stored in the property.
            value (id): value of the property.
            units: (None): units associated with the quantity's value
            tags (list<str>): list of strings storing metadata from
                Quantity evaluation.
            provenance (ProvenanceElement): provenance associated with the
                object (e. g. inputs, model, see ProvenanceElement)
        """

        self._value = value
        self._symbol_type = symbol_type
        self._tags = tags
        self._provenance = provenance
        self._internal_id = uuid.uuid4().hex

        # TODO: Move this to Model.evaluate()
        # if self._provenance is not None:
        #     if isinstance(self._provenance.source, dict):
        #         if 'source_key' not in self._provenance.source.keys() or \
        #                 self._provenance.source['source_key'] in (None, ""):
        #             self._provenance.source['source_key'] = self._internal_id
        #         if 'date_created' not in self._provenance.source.keys() or \
        #                 self._provenance.source['date_created'] in (None, ""):
        #             self._provenance.source['date_created'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #     elif self._provenance.source is None:
        #         self._provenance.source = {"source": "propnet",
        #                                    "source_key": self._internal_id,
        #                                    "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        if self._provenance is not None:
            if not isinstance(self._provenance.source, dict):
                self._provenance.source = {"source": self._provenance.source}

            if 'date_created' not in self._provenance.source.keys() or \
                    self._provenance.source['date_created'] in (None, ""):
                self._provenance.source['date_created'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            self._provenance = ProvenanceElement(source={"source": "",
                                                         "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    @staticmethod
    def _get_symbol_from_string(name):
        # Invoke default symbol if symbol is a string
        if not isinstance(name, str):
            raise TypeError("Expected str, encountered {}".format(type(name)))

        if name not in DEFAULT_SYMBOLS.keys():
            raise ValueError("Symbol type {} not recognized".format(name))

        return DEFAULT_SYMBOLS[name]

    # TODO: Do we need this method any more now that we have the factory?
    @staticmethod
    def to_quantity(symbol: Union[str, Symbol],
                    to_coerce: Union[float, np.ndarray, ureg.Quantity, "BaseQuantity"]) -> "BaseQuantity":
        """
        Converts the argument into the appropriate child object of a BaseQuantity object based on its type:
        - float -> given default units (as a pint object) and wrapped in a BaseQuantity object
        - numpy.ndarray array -> given default units (pint) and wrapped in a BaseQuantity object
        - ureg.Quantity -> simply wrapped in a BaseQuantity object
        - BaseQuantity -> immediately returned without modification
        - Any other python object -> simply wrapped in a BaseQuantity object

        TODO: Have a python object convert into an ObjectQuantity object or similar.

        Args:
            symbol: a string or Symbol object representing the type of data stored
            to_coerce: item to be converted into a BaseQuantity object
        Returns:
            (BaseQuantity) item that has been converted into a BaseQuantity object
        """
        # If a quantity is passed in, return the quantity.
        if isinstance(to_coerce, BaseQuantity):
            return to_coerce

        # Else
        # Convert the symbol to a Symbol if necessary.
        if isinstance(symbol, str):
            symbol = DEFAULT_SYMBOLS.get(symbol)
            if symbol is None:
                raise Exception("Attempted to create a quantity for an unrecognized symbol: " + str(symbol))
        # Return the correct BaseQuantity - warn if units are assumed.
        return Quantity(symbol, to_coerce)

    @classmethod
    def from_default(cls, symbol):
        """
        Class method to invoke a default quantity from a symbol name

        Args:
            symbol (Symbol or str): symbol or string corresponding to
                the symbol name

        Returns:
            BaseQuantity corresponding to default quantity from default
        """
        val = DEFAULT_SYMBOL_VALUES.get(symbol)
        if val is None:
            raise ValueError("No default value for {}".format(symbol))
        prov = ProvenanceElement(model='default', inputs=[])
        return Quantity(symbol, val, provenance=prov)

    @abstractmethod
    def magnitude(self):
        return

    @property
    def symbol(self):
        """
        Returns:
            (Symbol): Symbol of the BaseQuantity
        """
        return self._symbol_type

    @property
    def tags(self):
        """
        Returns:
            (list<str>): tags of the BaseQuantity
        """
        return self._tags

    @property
    def provenance(self):
        """
        Returns:
            (id): time of creation of the BaseQuantity
        """
        return self._provenance

    @property
    def value(self):
        """
        Returns:
            (id): value of the BaseQuantity
        """
        return self._value

    @abstractmethod
    def pretty_string(self):
        return

    def is_cyclic(self, visited=None):
        """
        Algorithm for determining if there are any cycles in
        the provenance tree, i. e. a repeated quantity in a
        tree branch

        Args:
            visited (list of visited model/symbols in the built tree
                that allows for recursion

        Returns:
            (bool) whether or not there is a cycle in the quantity
                provenance, i. e. repeated value in a tree branch
        """
        if visited is None:
            visited = set()
        if self.symbol in visited:
            return True
        visited.add(self.symbol)
        if self.provenance is None:
            return False
        # add distinct model hash to distinguish properties from models,
        # e.g. pugh ratio
        model_hash = "model_{}".format(self.provenance.model)
        if model_hash in visited:
            return True
        visited.add(model_hash)
        for p_input in self.provenance.inputs or []:
            this_visited = visited.copy()
            if p_input.is_cyclic(this_visited):
                return True
        return False

    def get_provenance_graph(self, start=None, filter_long_labels=True):
        """
        Gets an nxgraph object corresponding to the provenance graph

        Args:
            start (nxgraph): starting graph to build from

        Returns:
            (nxgraph): graph representation of provenance
        """
        graph = start or nx.MultiDiGraph()
        # import nose; nose.tools.set_trace()
        label = self.pretty_string()
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
            source = "Source: {}".format(source)
            graph.add_edge(source, self)

        return graph

    def draw_provenance_graph(self, filename, prog='dot',**kwargs):
        nx_graph = self.get_provenance_graph()
        a_graph = nx.nx_agraph.to_agraph(nx_graph)
        a_graph.node_attr['style'] = 'filled'
        a_graph.draw(filename, prog=prog, **kwargs)

    @abstractmethod
    def contains_nan_value(self):
        return

    @abstractmethod
    def contains_complex_type(self):
        return

    @abstractmethod
    def contains_imaginary_value(self):
        return

    def __hash__(self):
        return hash(self.symbol.name)

    def __eq__(self, other):
        if not isinstance(other, BaseQuantity) \
                or self.symbol != other.symbol \
                or self.symbol.category != other.symbol.category:
            return False
        return self.value == other.value

    def __str__(self):
        return "<{}, {}, {}>".format(self.symbol.name, self.value, self.tags)

    def __repr__(self):
        return self.__str__()

    def __bool__(self):
        return bool(self.value)


class NumQuantity(BaseQuantity):
    _ACCEPTABLE_SCALAR_TYPES = (int, float, complex)
    _ACCEPTABLE_ARRAY_TYPES = (list, np.ndarray)
    _ACCEPTABLE_DTYPES = (np.integer, np.floating, np.complexfloating)
    _ACCEPTABLE_TYPES = _ACCEPTABLE_SCALAR_TYPES + \
                        _ACCEPTABLE_ARRAY_TYPES + \
                        _ACCEPTABLE_DTYPES + \
                        (ureg.Quantity,)
    # This must be checked for explicitly because bool is a subtype of int
    # and isinstance(True/False, int) returns true
    _UNACCEPTABLE_TYPES = (bool,)

    def __init__(self, symbol_type, value, units=None, tags=None,
                 provenance=None, uncertainty=None):

        if isinstance(symbol_type, str):
            symbol_type = BaseQuantity._get_symbol_from_string(symbol_type)

        # Set default units if not supplied
        units = units or symbol_type.units

        if isinstance(value, self._ACCEPTABLE_DTYPES):
            value_in = ureg.Quantity(np.asscalar(value), units)
        elif isinstance(value, ureg.Quantity):
            value_in = value.to(units)
        elif isinstance(value, self._ACCEPTABLE_TYPES) and not \
                isinstance(value, self._UNACCEPTABLE_TYPES):
            value_in = ureg.Quantity(value, units)
        elif isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, self._ACCEPTABLE_DTYPES):
                value_in = ureg.Quantity(value, units)
            else:
                raise TypeError('Non-numerical numpy array passed to constructor: {}'.format(type(value)))

        # elif isinstance(value, NumQuantity):
        #     # We can't return a new object from __init__, so we're just going to
        #     # copy the attributes needed to the new object
        #     if symbol_type.name is not value._symbol_type.name:
        #         raise ValueError("Passed symbol_type '{}' does not match symbol_type of "
        #                          "passed value NumQuantity '{}'".format(self._symbol_type.name,
        #                                                                 value._symbol_type.name))
        #
        #     try:
        #         value_in = ureg.Quantity(value.magnitude, value.units)
        #         value_in = value_in.to(units)
        #     except Exception as ex:
        #         if type(ex).__name__ is "DimensionalityError":
        #             raise ValueError("Specified units {} incompatible"
        #                              "with symbol_type {}".format(units, symbol_type.name))
        #         else:
        #             raise ex
        #
        #     tags = tags or value._tags
        #     provenance = provenance or value._provenance
        else:
            raise TypeError('Cannot parse type passed to NumQuantity: {}'.format(type(value)))

        super(NumQuantity, self).__init__(symbol_type, value_in,
                                          tags=tags, provenance=provenance)

        if uncertainty is not None:
            if isinstance(uncertainty, self._ACCEPTABLE_DTYPES):
                self._uncertainty = ureg.Quantity(np.asscalar(uncertainty), units)
            elif isinstance(uncertainty, (float, int, list, complex)) and not \
                    isinstance(uncertainty, self._UNACCEPTABLE_TYPES):
                self._uncertainty = ureg.Quantity(uncertainty, units)
            elif isinstance(uncertainty, np.ndarray):
                if np.issubdtype(uncertainty.dtype, self._ACCEPTABLE_DTYPES):
                    self._uncertainty = ureg.Quantity(uncertainty, units)
                else:
                    raise TypeError('Non-numerical type passed to NumQuantity: {}'.format(type(uncertainty)))
            elif isinstance(uncertainty, ureg.Quantity):
                self._uncertainty = uncertainty.to(units)
            elif isinstance(uncertainty, NumQuantity):
                self._uncertainty = uncertainty._value.to(units)
            else:
                raise TypeError('Unknown type passed to NumQuantity: {}'.format(type(uncertainty)))
        else:
            self._uncertainty = None

        # TODO: Symbol-level constraints are hacked together atm,
        #       constraints as a whole need to be refactored and
        #       put into a separate module. They also are only
        #       available for numerical symbols because it uses
        #       sympy to evaluate the constraints. Would be better
        #       to make some class for symbol and/or model constraints
        if symbol_type.constraint is not None:
            if not symbol_type.constraint(**{symbol_type.name: self.magnitude}):
                raise SymbolConstraintError(
                    "NumQuantity with {} value does not satisfy {}".format(
                        value, symbol_type.constraint))

    # Not sure if we need this...
    # def __deepcopy__(self, memodict={}):
    #     return NumQuantity(self._symbol_type, self.value,
    #                        units=self.units, tags=self.tags,
    #                        provenance=self.provenance,
    #                        uncertainty=self._uncertainty)

    # TODO: Check to see if this works the same way as it did in old implementation
    def to(self, units):
        """
        Method to convert quantities between units, a la pint

        Args:
            units (tuple or str): units to convert quantity to

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
            quantities ([BaseQuantity]): list of quantities

        Returns:
            weighted mean
        """

        if not all(isinstance(q, cls) for q in quantities):
            # TODO: can't average ObjQuantities, highlights a weakness in
            # Quantity class that might be fixed by changing class design
            # ^^^ Not sure how we can address this
            raise ValueError("Weighted mean cannot be applied to objects")

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
        return self._value.magnitude

    @property
    def units(self):
        return self._value.units

    @property
    def uncertainty(self):
        return self._uncertainty

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
            if not hasattr(l, '__iter__'):
                raise TypeError("Type '{}' is not a list or is not iterable".format(type(l)))

            # Acceptable data types
            acceptable_dtypes = NumQuantity._ACCEPTABLE_DTYPES
            # acceptable_types = NumQuantity._ACCEPTABLE_TYPES + tuple([NumQuantity])   # Do we want to support copy?
            acceptable_types = NumQuantity._ACCEPTABLE_TYPES
            unacceptable_types = NumQuantity._UNACCEPTABLE_TYPES

            nested_lists = [v for v in l if isinstance(v, list)]
            np_arrays = [v for v in l if isinstance(v, np.ndarray)]
            ureg_quantities = [v for v in l if isinstance(v, ureg.Quantity)]
            regular_data = [v for v in l if not isinstance(v, (list, np.ndarray))]

            regular_data_is_type = all([isinstance(v, acceptable_types) and not
                                        isinstance(v, unacceptable_types)
                                        for v in regular_data])
            # If nested_lists is empty, all() returns true automatically
            nested_lists_is_type = all(recursive_list_type_check(v) for v in nested_lists)
            # issubdtype() accepting tuples as second arg is deprecated, using list comp instead
            np_arrays_is_type = all(any(np.issubdtype(v.dtype, dt) for dt in acceptable_dtypes)
                                    for v in np_arrays)

            ureg_quantities_is_type = all(recursive_list_type_check([v.magnitude])
                                          for v in ureg_quantities)

            return regular_data_is_type and nested_lists_is_type and np_arrays_is_type

        return recursive_list_type_check([to_check])

    def pretty_string(self, sigfigs=4):
        # TODO: maybe support a rounding kwarg?
        if isinstance(self.magnitude, self._ACCEPTABLE_SCALAR_TYPES):
            out = "{1:.{0}g}".format(sigfigs, self.magnitude)
            if self.uncertainty:
                out += "\u00B1{0:.4g}".format(self.uncertainty.magnitude)
        else:
            out = "{}".format(self.magnitude)

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


class ObjQuantity(BaseQuantity):
    def __init__(self, symbol_type, value, tags=None,
                 provenance=None):
        if isinstance(symbol_type, str):
            symbol_type = super()._get_symbol_from_string(symbol_type)
        super(ObjQuantity, self).__init__(symbol_type, value, tags=tags, provenance=provenance)

    @property
    def magnitude(self):
        return self._value

    def pretty_string(self):
        return "{}".format(self.value)

    # TODO: Determine whether it's necessary to define these for ObjQuantity
    # we could just assess this if models return NumQuantity
    def contains_nan_value(self):
        return False

    def contains_complex_type(self):
        return False

    def contains_imaginary_value(self):
        return False


def Quantity(symbol_type, value, units=None, tags=None,
             provenance=None, uncertainty=None):
    """
    Factory method for BaseQuantity class, provides more intuitive call
    to create new BaseQuantity child objects and provide backwards
    compatibility.

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
    if NumQuantity.is_acceptable_type(value):
        return NumQuantity(symbol_type, value,
                           units=units, tags=tags,
                           provenance=provenance,
                           uncertainty=uncertainty)

    if units is not None:
        logger.warning("Cannot assign units to value of type '{}'."
                       "Ignoring units.".format(type(value)))

    return ObjQuantity(symbol_type, value,
                       tags=tags,
                       provenance=provenance)


class StorageQuantity(MSONable):
    def __init__(self, data_type=None, symbol_type=None,
                 internal_id=None, tags=None, provenance=None):

        self._internal_id = internal_id
        self._data_type = data_type
        self._symbol_type = symbol_type
        self._tags = tags
        self._provenance = provenance

    @classmethod
    def from_quantity(cls, quantity_in):
        if isinstance(quantity_in, StorageQuantity):
            return copy.deepcopy(quantity_in)
        elif issubclass(quantity_in, BaseQuantity):
            data_type = type(quantity_in).__name__
        else:
            raise TypeError("Expected StorageQuantity or"
                            "object that inherits BaseQuantity, instead received {}"
                            .format(type(quantity_in)))

        return cls.__init__(data_type=data_type, symbol_type=quantity_in._symbol_type,
                            internal_id=quantity_in._internal_id, tags=quantity_in.tags,
                            provenance=cls._convert_provenance_for_storage(quantity_in.provenance))

    @staticmethod
    def _convert_provenance_for_storage(provenance_in):
        provenance_out = copy.deepcopy(provenance_in)
        if provenance_in.inputs is not None:
            provenance_out.inputs = [StorageQuantity.from_quantity(q)
                                     for q in provenance_in.inputs]

        return provenance_out

    def to_quantity(self):
        raise NotImplementedError('Still working on this feature!')
