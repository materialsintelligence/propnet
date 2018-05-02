import numpy as np

from typing import *
from uuid import UUID
from monty.json import MSONable

from propnet import logger, ureg
from propnet.core.symbols import Symbol


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

        # TODO: move Quantity + Symbol to separate files to remove circular import
        from propnet.symbols import DEFAULT_SYMBOL_TYPES
        if isinstance(symbol_type, str):
            if symbol_type not in DEFAULT_SYMBOL_TYPES.keys():
                raise ValueError("Quantity type {} not recognized".format(symbol_type))
            symbol_type = DEFAULT_SYMBOL_TYPES[symbol_type]

        if type(value) == float or type(value) == int:
            value = ureg.Quantity(value, symbol_type.units)
        elif type(value) == ureg.Quantity:
            value = value.to(symbol_type.units)

        self._symbol_type = symbol_type
        self._value = value
        self._tags = tags
        self._provenance = provenance

    # Associated accessor methods.
    @property
    def value(self):
        """
        Returns:
            (id): value of the Quantity
        """
        return self._value

    @property
    def type(self):
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
        return hash(self.type.name)

    def __eq__(self, other):
        if not isinstance(other, Quantity):
            return False
        if self.type != other.type:
            return False
        if type(self.value) == ureg.Quantity:
            val1 = float(self.value.magnitude)
        else:
            val1 = self.value
        if type(other.value) == ureg.Quantity:
            val2 = float(other.value.magnitude)
        else:
            val2 = other.value
        if not np.isclose(val1, val2):
            return False
        return True

    def __str__(self):
        return "<{}, {}, {}>".format(self.type.name, self.value, self.tags)