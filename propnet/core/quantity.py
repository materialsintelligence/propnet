# TODO: remove typing?
from typing import Union, Any, Optional, List

import numpy as np
from monty.json import MSONable

from propnet import ureg
from propnet.core.symbols import Symbol
from propnet.core.provenance import ProvenanceElement
from propnet.symbols import DEFAULT_SYMBOLS
from uncertainties import unumpy

from propnet.core.exceptions import SymbolConstraintError


#TODO: Generally the propnet Quantity/pint Quantity distinction is not
#      very facile, I think this can be streamlined a bit
class Quantity(MSONable):
    """
    Class storing the value of a property.

    Constructed by the user to assign values to abstract Symbol types.
    Represents the fact that a given Quantity has a given value. They
    are added to the PropertyNetwork graph in the context of Material
    objects that store collections of Quantity objects representing
    that a given material has those properties.

    Attributes:
        symbol_type: (Symbol) the type of information that is represented
            by the associated value.
        value: (id) the value associated with this symbol.
        tags: (list<str>) tags associated with the material, e.g.
            perovskites or ferroelectrics
    """

    def __init__(self,
                 symbol_type: Union[str, Symbol],
                 value: Any,
                 tags: Optional[List[str]]=None,
                 provenance=None):
        """
        Parses inputs for constructing a Property object.

        Args:
            symbol_type (Symbol): pointer to an existing PropertyMetadata
                object or String giving the name of a SymbolType object,
                identifies the type of data stored in the property.
            value (id): value of the property.
            tags (list<str>): list of strings storing metadata from
                Quantity evaluation.
            provenance (ProvenanceElement): provenance associated with the
                object (e. g. inputs, model, see ProvenanceElement)
        """

        if isinstance(symbol_type, str):
            if symbol_type not in DEFAULT_SYMBOLS.keys():
                raise ValueError("Quantity type {} not recognized".format(symbol_type))
            symbol_type = DEFAULT_SYMBOLS[symbol_type]

        if type(value) == float or type(value) == int:
            value = ureg.Quantity(value, symbol_type.units)
        elif type(value) == ureg.Quantity:
            value = value.to(symbol_type.units)

        if symbol_type.constraint is not None:
            if not symbol_type.constraint.subs({symbol_type.name: value.magnitude}):
                raise SymbolConstraintError(
                    "Quantity with {} value does not satisfy {}".format(
                        value, symbol_type.constraint))

        self._symbol_type = symbol_type
        self._value = value
        self._tags = tags
        self._provenance = provenance

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

    def is_cyclic(self, visited=None):
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

        # TODO: test this with units, not magnitudes ... remember units
        # may not be canonical units(?)
        if isinstance(quantities[0].value, list):
            # hack to get arrays working for now
            vals = [q.value for q in quantities]
        else:
            vals = [q.value.magnitude for q in quantities]

        new_magnitude = np.mean(vals, axis=0)
        std_dev = np.std(vals, axis=0)
        new_value = unumpy.uarray(new_magnitude, std_dev)

        new_tags = set()
        new_provenance = ProvenanceElement(model='aggregation', inputs=[])
        for quantity in quantities:
            if quantity.tags:
                for tag in quantity.tags:
                    new_tags.add(tag)
            new_provenance.inputs.append(quantity)

        return cls(symbol_type=input_symbol, value=new_value,
                   tags=list(new_tags), provenance=new_provenance)
