"""
Module containing classes and methods for Material functionality in propnet code.
"""

from collections import defaultdict
from itertools import chain

from propnet.core.quantity import QuantityFactory, NumQuantity
from propnet.core.symbols import Symbol

# noinspection PyUnresolvedReferences
import propnet.symbols
from propnet.core.registry import Registry
import logging

logger = logging.getLogger(__name__)


class Material(object):
    """
    Class containing methods for creating and interacting with Material objects.

    Under the Propnet infrastructure, Materials are the medium through which properties are
    communicated. While Model and Symbol nodes create a web of interconnected properties,
    Materials, as collections of Quantity nodes, provide concrete numbers to those properties.
    At runtime, a Material can be constructed and added to a Graph instance, merging the two
    graphs and allowing for propagation of concrete numbers through the property web.

    A unique hashcode is stored with each Material upon instantiation. This is used to
    differentiate between different materials at runtime.

    Attributes:
        symbol_quantities_dict (dict<Symbol, set<Quantity>>): data structure mapping Symbols to a list of corresponding
                                                           Quantity objects of that type.

    """
    def __init__(self, quantities=None, add_default_quantities=False):
        """
        Creates a Material instance, instantiating a trivial graph of one node.

        Args:
            quantities ([Quantity]): list of quantities to add to
                the material
            add_default_quantities (bool): whether to add default
                quantities (e. g. room temperature) to the graph
        """
        self._quantities_by_symbol = defaultdict(set)
        if quantities is not None:
            for quantity in quantities:
                self.add_quantity(quantity)

        if add_default_quantities:
            self.add_default_quantities()

    def add_quantity(self, quantity):
        """
        Adds a property to this property collection.

        Args:
            quantity (Quantity): property to be bound to the material.

        Returns:
            None
        """
        self._quantities_by_symbol[quantity.symbol].add(quantity)

    def remove_quantity(self, quantity):
        """
        Removes the Quantity object attached to this Material.

        Args:
            quantity (Quantity): Quantity object reference indicating
            which property is to be removed from this Material.

        Returns:
            None
        """
        if quantity.symbol not in self._quantities_by_symbol:
            raise Exception("Attempting to remove quantity not present in "
                            "the material.")
        self._quantities_by_symbol[quantity.symbol].remove(quantity)

    def add_default_quantities(self):
        """
        Adds any default symbols which are not present in the graph

        Returns:
            None
        """
        new_syms = set(Registry("symbol_values").keys())
        new_syms -= set(self._quantities_by_symbol.keys())
        for sym in new_syms:
            quantity = QuantityFactory.from_default(sym)
            logger.warning("Adding default {} quantity with value {}".format(
                           sym, quantity))
            self.add_quantity(quantity)

    def remove_symbol(self, symbol):
        """
        Removes all Quantity Nodes attached to this Material of type symbol.

        Args:
            symbol (Symbol): object indicating which property type
                is to be removed from this material.

        Returns:
            None
        """
        if symbol not in self._quantities_by_symbol:
            raise Exception("Attempting to remove Symbol not present in the material.")
        del self._quantities_by_symbol[symbol]

    def get_symbols(self):
        """
        Obtains all Symbol objects bound to this Material.

        Returns:
            (set<Symbol>) set containing all symbols bound to this Material.
        """
        return set(self._quantities_by_symbol.keys())

    def get_quantities(self):
        """
        Method obtains all Quantity objects bound to this Material.
        Returns:
            (list<Quantity>) list of all Quantity objects bound to this Material.
        """
        return list(chain.from_iterable(self._quantities_by_symbol.values()))
    
    @property
    def symbol_quantities_dict(self):
        return self._quantities_by_symbol.copy()
    
    def get_aggregated_quantities(self):
        """
        Return mean values for all quantities for each symbol.

        Returns:
            (dict<Symbol, weighted_mean) mapping from a Symbol to
            an aggregated statistic.
        """
        # TODO: proper weighting system, and more flexibility in object handling
        aggregated = {}
        for symbol, quantities in self._quantities_by_symbol.items():
            if not symbol.category == 'object':
                aggregated[symbol] = NumQuantity.from_weighted_mean(list(quantities))
        return aggregated

    def __str__(self):
        QUANTITY_LENGTH_CAP = 50
        building = []
        building += ["Material: " + str(hex(id(self))), ""]
        for symbol in self._quantities_by_symbol.keys():
            building += ["\t" + symbol.name]
            for quantity in self._quantities_by_symbol[symbol]:
                qs = str(quantity)
                if "\n" in qs or len(qs) > QUANTITY_LENGTH_CAP:
                    qs = "..."
                building += ["\t\t" + qs]
            building += [""]
        return "\n".join(building)

    def __eq__(self, other):
        if not isinstance(other, Material):
            return False
        if len(self._quantities_by_symbol) != len(other._quantities_by_symbol):
            return False
        for symbol in self._quantities_by_symbol.keys():
            if symbol not in other._quantities_by_symbol.keys():
                return False
            if len(self._quantities_by_symbol[symbol]) != len(other._quantities_by_symbol[symbol]):
                return False
            for quantity in self._quantities_by_symbol[symbol]:
                if quantity not in other._quantities_by_symbol[symbol]:
                    return False
        return True

    @property
    def quantity_types(self):
        return list(self._quantities_by_symbol.keys())

    def __getitem__(self, item):
        return self._quantities_by_symbol[item]


class CompositeMaterial(Material):
    """
    Class representing a material composed of one or more sub-materials.

    Useful for representing materials properties that arise from
    multiple materials (i. e. contact voltage in metals)

    Attributes:
        symbol_quantities_dict (dict<Symbol, set<Quantity>>): data-structure
            storing all properties / descriptors that arise from the
            joining of multiple materials
        materials (list<Material>): set of materials contained in the Composite
    """
    def __init__(self, materials_list):
        """
        Creates a Composite Material instance.

        Args:
            materials_list (list<Material>): list of materials contained
                in the Composite
        """
        self.materials = materials_list
        super(CompositeMaterial, self).__init__()
