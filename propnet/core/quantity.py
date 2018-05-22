import numpy as np

from typing import *
from uuid import UUID
from monty.json import MSONable
from uncertainties import unumpy

from propnet import logger, ureg
from propnet.core.symbols import Symbol
from propnet.symbols import DEFAULT_SYMBOLS

class Quantity(MSONable):
    """
    Class storing the value of a property.

    Constructed by the user to assign values to abstract Symbol types. Represents the fact
    that a given Quantity
    has a given value. They are added to the PropertyNetwork graph in the context of Material
    objects that store
    collections of Quantity objects representing that a given material has those properties.

    Attributes:
        type: (Symbol) the type of information that is represented by the associated value.
        value: (id) the value associated with this symbol.
        tags: (list<str>)
    """

    def __init__(self,
                 symbol_type: Union[str, Symbol],
                 value: Any,
                 tags: Optional[List[str] ] =None,
                 material: Union[str, UUID ] =None,
                 sources: List= None,
                 provenance=None):
        """
        Parses inputs for constructing a Property object.

        Args:
            symbol_type (Symbol): pointer to an existing PropertyMetadata object or String
            giving
                    the name of a SymbolType object, identifies the type of data stored
                    in the property.
            value (id): value of the property.
            tags (list<str>): list of strings storing metadata from Quantity evaluation.
            provenance (id): time of creation of the object.
        """

        if isinstance(symbol_type, str):
            if symbol_type not in DEFAULT_SYMBOLS.keys():
                raise ValueError("Quantity type {} not recognized".format(symbol_type))
            symbol_type = DEFAULT_SYMBOLS[symbol_type]

        if type(value) == float or type(value) == int:
            value = ureg.Quantity(value, symbol_type.units)
        elif type(value) == ureg.Quantity:
            value = value.to(symbol_type.units)

        self._symbol_type = symbol_type
        self._value = value
        self._tags = tags
        self._provenance = provenance

    #def as_dict(self):
    #    return {
    #        "@module": "propnet.core.quantity",
    #        "@class": "Quantity",
    #        "symbol": self.s#ymbol.name
    #    }

    @property
    def value(self):
        """
        Returns:
            (id): value of the Quantity
        """
        return self._value

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


def weighted_mean(quantities: List[Quantity]) -> List[Quantity]:

    # can't run this twice yet ...
    # TODO: remove
    if hasattr(quantities[0].value, "std_dev"):
        return quantities

    input_symbol = quantities[0].symbol
    if input_symbol.category == 'object':
        # TODO: can't average 'objects', highlights a weakness in Quantity class
        # would be fixed by changing this class design ...
        return quantities

    if not all(input_symbol == q.symbol for q in quantities):
        raise ValueError("Can only calculate a weighted mean if all quantities "
                         "refer to the same symbol.")

    # TODO: an actual weighted mean; just a simple mean at present
    # TODO: support propagation of uncertainties (this will only work once at present)

    # TODO: test this with units, not magnitudes ... remember units may not be canonical units(?)
    vals = [float(q.value.magnitude) for q in quantities]
    new_magnitude = np.mean(vals, axis=0)
    std_dev = np.std(vals, axis=0)
    new_value = unumpy.uarray(new_magnitude, std_dev)

    new_tags = set()
    new_provenance = set()
    for q in quantities:
        if q.tags:
            for t in q.tags:
                new_tags.add(t)
        if q.provenance:
            for p in q.provenance:
                new_provenance.add(p)

    new_quantity = Quantity(symbol_type=input_symbol,
                            value=new_value,
                            tags=list(new_tags),
                            provenance=list(new_provenance))
    print(new_quantity)

    return new_quantity