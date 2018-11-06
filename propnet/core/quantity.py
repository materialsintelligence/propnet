import numpy as np
from monty.json import MSONable
from collections import OrderedDict

from scipy.optimize import minimize

from propnet import ureg
from propnet.core.symbols import Symbol
from propnet.core.provenance import ProvenanceElement
from propnet.symbols import DEFAULT_SYMBOLS, DEFAULT_SYMBOL_VALUES

from propnet.core.exceptions import SymbolConstraintError


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
            symbol_type (Symbol): pointer to an existing Symbol
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
        if isinstance(value, (float, int, list, np.ndarray)):
            self._value = ureg.Quantity(value, units)
        elif isinstance(value, ureg.Quantity):
            self._value = value.to(units)
        else:
            self._value = value

        if isinstance(uncertainty, (float, int, list, np.ndarray)):
            self._uncertainty = ureg.Quantity(value, units)
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

    @property
    def is_pint(self):
        return isinstance(self._value, ureg.Quantity)

    def pretty_string(self, sigfigs=4):
        if self.is_pint:
            if isinstance(self.magnitude, (float, int)):
                out = "{0:.4g}".format(self.magnitude)
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
        for input in self.provenance.inputs:
            this_visited = visited.copy()
            if input.is_cyclic(this_visited):
                return True
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


# TODO: These fitting utils should likely go in a separate package
#       that's devoted to the data-science oriented side of propnet
# TODO: this is very preliminary, could be improved substantially
def aggregate_quantities(quantities, model_score_dict=None):
    """
    Simple method for aggregating a set of quantities

    Args:
        quantities:
        model_score_dict:

    Returns:

    """
    if not all([q.symbol == quantities[0].symbol for q in quantities]):
        raise ValueError("Quantities passed to aggregate must be same symbol")
    weights = [get_weight(q, model_score_dict) for q in quantities]
    result_value = sum(
        [w * q.value for w, q in zip(weights, quantities)]) / sum(weights)
    return Quantity(quantities[0].symbol, result_value)


def get_weight(quantity, model_score_dict=None):
    """
    Gets weight based on scoring scheme

    Args:
        quantity (Quantity): quantity for which to get weight
        model_score_dict ({str: float}): dictionary of model names to scores

    Returns:
        calculated weight for input quantity
    """
    if quantity.provenance is None:
        return 1
    if model_score_dict is None:
        return 1
    weight = model_score_dict.get(quantity.provenance.model)
    weight *= np.prod(
        [get_weight(q, model_score_dict) for q in quantity.provenance.inputs])
    return weight


# TODO: Add default graph when this is moved
def fit_model_scores(materials, benchmarks, models=None,
                     min_score=0.05, init_scores=None):
    """
    Fits a set of model scores to a set of benchmark data

    Args:
        materials ([Material]): list of materials
        benchmarks ([{Symbol or str: float}]): list of benchmarks,
            containing
        models ([Model or str]): list of models which should have their
            scores adjusted in the aggregation weighting scheme
        min_score (float): minimum score to use in weighting
            scheme, defaults to 0.05
        init_scores ({str: float}): scores to initialize minimize
            procedure with

    Returns:
        {str: float} scores corresponding to those which minimize
            SSE for the benchmarked dataset

    """
    # graph = graph #or Graph() Add default graph when this is moved
    # model_list = graph._models.keys()
    models = DEFAULT_MODEL_NAMES
    def f(scores):
        model_score_dict = {m: max(s, min_score)
                            for m, s in zip(model_list, scores)}
        return get_sse(materials, benchmarks, model_score_dict)
    scores = OrderedDict((m, 1) for m in model_list)
    scores.update(init_scores or {})
    result = minimize(f, x0=np.array(scores.values()))
    vec = [max(s, min_score) for s in result.x]
    return OrderedDict(zip(model_list, vec))


def get_sse(materials, benchmarks, model_score_dict=None):
    """
    Function to get the sum squared error of a set of benchmarks
    with aggregated data from the model scoring scheme above

    Args:
        materials ([Material]): list of materials to evaluate
        benchmarks ([{Symbol or str: float}]): list of benchmarks
            for each material
        model_score_dict ({str: float}): model score dictionary
            with scores for each model name

    Returns:
        (float): sum squared error over all the benchmarks

    """
    sse = 0
    for material, benchmark in zip(materials, benchmarks):
        for symbol, value in benchmark.items():
            agg = aggregate_quantities(material[symbol], model_score_dict)
            sse += (agg.magnitude - benchmark[symbol]) ** 2
    return sse
