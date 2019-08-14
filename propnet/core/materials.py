"""Materials objects to hold properties of materials for evaluation.

This module establishes objects to represent single (Material) and mixed (CompositeMaterial) materials.
They are effectively containers for their properties (and processing conditions). The material objects
are the expected inputs for propnet's Graph class.

Example:
    Material objects can be instantiated empty or with a list of Quantity objects representing the
    material's properties:

    >>> from propnet.core.materials import Material
    >>> from propnet.core.quantity import QuantityFactory as QF
    >>> band_gap = QF.create_quantity('band_gap', 5, 'eV')
    >>> bulk_modulus = QF.create_quantity('bulk_modulus', 100, 'GPa')
    >>> m = Material([band_gap, bulk_modulus]) # Initialize with list, or...
    >>> m = Material()  # Initialize empty and add properties
    >>> m.add_quantity(band_gap)
    >>> m.add_quantity(bulk_modulus)

"""

import logging
from collections import defaultdict
from itertools import chain

from propnet.core.quantity import QuantityFactory, NumQuantity, BaseQuantity
from propnet.core.symbols import Symbol
from propnet.core.registry import Registry

logger = logging.getLogger(__name__)
"""logging.Logger: Logger for debugging"""


class Material:
    """
    Class containing methods to interact with materials with a single composition. This class is intended to
    be a container for materials properties.

    Examples:
        The example shown above largely demonstrates the utility of this class. However, it is worth noting
        that a Material object can be accessed like a dictionary keyed by Symbol objects to retrieve the
        set of quantities that correspond to that symbol.

        >>> m = Material([...])
        >>> quantities = m['band_gap']
        >>> print(quantities)
        {<band_gap, ...>, ...}

    """

    def __init__(self, quantities=None, add_default_quantities=False):
        """
        Args:
            quantities (`list` of `BaseQuantity` or `None`): optional, list of quantities to add to
                the material. Default: ``None`` (no properties added)
            add_default_quantities (bool): ``True`` adds default quantities (e.g. room temperature)
                to the graph. ``False`` omits them. Default quantities are defined as Symbols who have
                a default value specified and are registered in ``Registry('symbol_values')``.
                Default: ``False`` (omit default quantities)
        """
        self._quantities_by_symbol = defaultdict(set)
        if quantities is not None:
            for quantity in quantities:
                self.add_quantity(quantity)

        if add_default_quantities:
            self.add_default_quantities()

    def add_quantity(self, quantity):
        """
        Adds a property to this material.

        Args:
            quantity (BaseQuantity): property to be bound to the material.
        """
        self._quantities_by_symbol[quantity.symbol].add(quantity)

    def remove_quantity(self, quantity):
        """
        Removes a quantity attached to this Material.

        Args:
            quantity (BaseQuantity): reference to quantity object to be removed
        """
        if quantity.symbol not in self._quantities_by_symbol:
            raise KeyError("Attempting to remove quantity not present in "
                           "the material.")
        self._quantities_by_symbol[quantity.symbol].remove(quantity)

    def add_default_quantities(self):
        """
        Adds any default symbols which are not present in the graph. Default symbols
        are sourced from ``Registry('symbol_values')``.
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
        Removes all quantities attached to this material of a particular Symbol type.

        Args:
            symbol (Symbol): symbol to be removed from the material
        """
        if symbol not in self._quantities_by_symbol:
            raise KeyError("Attempting to remove Symbol not present in the material.")
        del self._quantities_by_symbol[symbol]

    def get_symbols(self):
        """
        Obtains all Symbol types bound to this material.

        Returns:
            `set` of `propnet.core.symbols.Symbol`: set containing all symbols bound to this material
        """
        return set(self._quantities_by_symbol.keys())

    def get_quantities(self):
        """
        Obtains all quantity objects bound to this material.

        Returns:
            `list` of `propnet.core.quantity.BaseQuantity`: list of all quantity objects bound to this material
        """
        return list(chain.from_iterable(self._quantities_by_symbol.values()))
    
    @property
    def symbol_quantities_dict(self):
        """
        dict: mapping of Symbols to the set of quantities of that Symbol type attached to the material
        """
        # TODO: This may not be safe enough. Might need deep copy.
        return self._quantities_by_symbol.copy()
    
    def get_aggregated_quantities(self):
        """
        Aggregates multiple quantities of the same symbol by calculating their mean. Does not mutate this
        Material object.

        Returns:
            dict: dictionary mapping numerical Symbols to their aggregated mean value as ``NumQuantity`` objects.
        """
        # TODO: proper weighting system, and more flexibility in object handling
        aggregated = {}
        for symbol, quantities in self._quantities_by_symbol.items():
            if not symbol.category == 'object':
                aggregated[symbol] = NumQuantity.from_weighted_mean(list(quantities))
        return aggregated

    def __str__(self):
        """
        Builds summary of the material object including some properties.

        Returns:
            str: string representation of material
        """
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
        """
        Equality function for material. Two materials are equal if they have the same quantities.

        Args:
            other (object): the object to compare the Material to

        Returns:
            bool: ``True`` if equal, ``False`` if not or not the same type
        """
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
        """
        list: the Symbol objects contained in this material
        """
        return list(self._quantities_by_symbol.keys())

    def __getitem__(self, item):
        """
        Retrieve quantities in material by Symbol or symbol name.

        Args:
            item (`str` or `Symbol`): symbol to retrieve

        Returns:
            set: set containing the quantities attached to the material of the specified Symbol
        """
        return self._quantities_by_symbol.get(item, set())


class CompositeMaterial(Material):
    """
    Class representing a material composed of one or more sub-materials.

    Useful for representing materials properties that arise from
    multiple materials (i.e. contact voltage in metals).

    Notes:
        The functionality of this class is limited, but expansion is planned.

    Attributes:
        materials (`list` of `Material`): list of materials contained in the CompositeMaterial
    """
    def __init__(self, materials):
        """
        Args:
            materials (`list` of `Material`): list of materials contained
                in the CompositeMaterial
        """
        self.materials = materials
        super(CompositeMaterial, self).__init__()
