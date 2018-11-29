import numpy as np
from monty.json import MSONable

import networkx as nx

from propnet import ureg
from propnet.core.symbols import Symbol
from propnet.core.provenance import ProvenanceElement
from propnet.symbols import DEFAULT_SYMBOLS, DEFAULT_SYMBOL_VALUES

from propnet.core.exceptions import SymbolConstraintError
from typing import Union


def pint_only(f):
    """
    Decorator for methods or properties that should raise an error
    if the value is not a pint quantity
    """
    def wrapper(self, *args, **kwargs):
        if not self.is_pint:
            raise ValueError("{} only implemented for pint quantities".format(
                f.__name__))
        return f(self, *args, **kwargs)
    return wrapper


class Quantity(MSONable):
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

    def __init__(self, symbol_type, value, units=None, tags=None,
                 provenance=None, uncertainty=None):
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
        # Invoke default symbol if symbol is a string
        if isinstance(symbol_type, str):
            if symbol_type not in DEFAULT_SYMBOLS.keys():
                raise ValueError("Quantity type {} not recognized".format(symbol_type))
            symbol_type = DEFAULT_SYMBOLS[symbol_type]

        # Set default units if not supplied
        units = units or symbol_type.units

        # Invoke pint quantity if supplied or input is float/int

        if isinstance(value, (np.floating, np.integer, np.complexfloating)):
            self._value = ureg.Quantity(np.asscalar(value), units)
        elif isinstance(value, (float, int, list, complex, np.ndarray)):
            self._value = ureg.Quantity(value, units)
        elif isinstance(value, ureg.Quantity):
            self._value = value.to(units)
        elif isinstance(value, Quantity):
            self._value = value._value
        else:
            self._value = value

        if isinstance(uncertainty, (np.floating, np.integer, np.complexfloating)):
            self._uncertainty = ureg.Quantity(np.asscalar(uncertainty), units)
        elif isinstance(uncertainty, (float, int, list, complex, np.ndarray)):
            self._uncertainty = ureg.Quantity(uncertainty, units)
        elif isinstance(uncertainty, ureg.Quantity):
            self._uncertainty = uncertainty.to(units)
        else:
            self._uncertainty = uncertainty

        # TODO: Symbol-level constraints are hacked together atm,
        #       constraints as a whole need to be refactored and
        #       put into a separate module
        if symbol_type.constraint is not None:
            if not symbol_type.constraint(**{symbol_type.name: self.magnitude}):
                raise SymbolConstraintError(
                    "Quantity with {} value does not satisfy {}".format(
                        value, symbol_type.constraint))

        self._symbol_type = symbol_type
        self._tags = tags
        self._provenance = provenance

    @staticmethod
    def to_quantity(symbol: Union[str, Symbol],
                    to_coerce: Union[float, np.ndarray, ureg.Quantity, "Quantity"]) -> "Quantity":
        """
        Converts the argument into a Quantity object based on its type:
        - float -> given default units (as a pint object) and wrapped in a Quantity object
        - numpy.ndarray array -> given default units (pint) and wrapped in a Quantity object
        - ureg.Quantity -> simply wrapped in a Quantity object
        - Quantity -> immediately returned without modification
        - Any other python object -> simply wrapped in a Quantity object

        TODO: Have a python object convert into an ObjectQuantity object or similar.

        Args:
            to_coerce: item to be converted into a Quantity object
        Returns:
            (Quantity) item that has been converted into a Quantity object
        """
        # If a quantity is passed in, return the quantity.
        if isinstance(to_coerce, Quantity):
            return to_coerce

        # Else
        # Convert the symbol to a Symbol if necessary.
        if isinstance(symbol, str):
            symbol = DEFAULT_SYMBOLS.get(symbol)
            if symbol is None:
                raise Exception("Attempted to create a quantity for an unrecognized symbol: " + str(symbol))
        # Return the correct Quantity - warn if units are assumed.
        if isinstance(to_coerce, float) or isinstance(to_coerce, np.ndarray):
            return Quantity(symbol, ureg.Quantity(to_coerce, symbol.units))

        return Quantity(symbol, to_coerce)


    @property
    def is_pint(self):
        return isinstance(self._value, ureg.Quantity)

    def pretty_string(self, sigfigs=4):
        # TODO: maybe support a rounding kwarg?
        if self.is_pint:
            if isinstance(self.magnitude, (float, int)):
                out = "{1:.{0}g}".format(sigfigs, self.magnitude)
                if self.uncertainty:
                    out += "\u00B1{0:.4g}".format(self.uncertainty.magnitude)
            else:
                out = "{}".format(self.magnitude)
            out += " {:~P}".format(self.units)
        else:
            out = "{}".format(self.value)
        return out

    @property
    def value(self):
        """
        Returns:
            (id): value of the Quantity
        """
        return self._value

    @property
    @pint_only
    def magnitude(self):
        return self._value.magnitude

    @property
    def uncertainty(self):
        return self._uncertainty

    @pint_only
    def to(self, units):
        """
        Method to convert quantities between units, a la pint

        Args:
            units (tuple or str): units to convert quantity to

        Returns:

        """
        return Quantity(self.symbol, self._value, units, self.tags,
                        self._provenance, self.uncertainty)

    @property
    @pint_only
    def units(self):
        return self._value.units

    @property
    def symbol(self):
        """
        Returns:
            (Symbol): Symbol of the Quantity
        """
        return self._symbol_type

    @property
    def tags(self):
        """
        Returns:
            (list<str>): tags of the Quantity
        """
        return self._tags

    @property
    def provenance(self):
        """
        Returns:
            (id): time of creation of the Quantity
        """
        return self._provenance

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
        # Assumes all non-pint Quantity objects have non-numerical values, and therefore cannot be NaN
        if not self.is_pint:
            return False

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
        # Assumes all non-pint Quantity objects have non-numerical values, and therefore cannot be complex
        if not self.is_pint:
            return False

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
        elif isinstance(value, Quantity):
            return value.contains_complex_type()

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

    def __hash__(self):
        return hash(self.symbol.name)

    def __eq__(self, other):
        if not isinstance(other, Quantity) \
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

    # TODO: lazily implemented, fix to be a bit more robust
    def as_dict(self):
        if isinstance(self.value, ureg.Quantity):
            value = self.value.magnitude
            units = self.value.units
        else:
            value = self.value
            units = None
        return {"symbol_type": self._symbol_type.name,
                "value": value,
                "provenance": self._provenance,
                "units": units.format_babel() if units else None,
                "@module": "propnet.core.quantity",
                "@class": "Quantity"}

    @classmethod
    def from_default(cls, symbol):
        """
        Class method to invoke a default quantity from a symbol name

        Args:
            symbol (Symbol or str): symbol or string corresponding to
                the symbol name

        Returns:
            Quantity corresponding to default quantity from default
        """
        val = DEFAULT_SYMBOL_VALUES.get(symbol)
        if val is None:
            raise ValueError("No default value for {}".format(symbol))
        prov = ProvenanceElement(model='default', inputs=[])
        return cls(symbol, val, provenance=prov)

    @classmethod
    def from_weighted_mean(cls, quantities):
        """
        Function to invoke weighted mean quantity from other
        quantities

        Args:
            quantities ([Quantity]): list of quantities

        Returns:
            weighted mean
        """
        input_symbol = quantities[0].symbol
        if input_symbol.category == 'object':
            # TODO: can't average 'objects', highlights a weakness in
            # Quantity class that might be fixed by changing class design
            raise ValueError("Weighted mean cannot be applied to objects")

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
        std_dev = (sum([(v - new_value)**2 for v in vals]) / len(vals))**(1/2)

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
            graph.add_edge(source, self)

        return graph

    def draw_provenance_graph(self, filename, prog='dot',**kwargs):
        nx_graph = self.get_provenance_graph()
        a_graph = nx.nx_agraph.to_agraph(nx_graph)
        a_graph.node_attr['style'] = 'filled'
        a_graph.draw(filename, prog=prog, **kwargs)
