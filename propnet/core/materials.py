"""
Module containing classes and methods for Material functionality in propnet code.
"""

import networkx as nx
from itertools import chain

from collections import defaultdict

from propnet.core.symbols import Symbol
from propnet.core.quantity import Quantity, weighted_mean
from propnet.core.utils import uuid
from propnet.core.graph import Graph


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
        uuid (int): unique hash number used as an identifier for this object.
        parent (Graph): Stores a pointer to the Graph instance this Material has been bound to.
        _symbol_to_quantity (dict<Symbol, set<Quantity>>): data structure mapping Symbols to a list of corresponding
                                                           Quantity objects of that type.

    """
    def __init__(self):
        """
        Creates a Material instance, instantiating a trivial graph of one node.
        """
        self.uuid = uuid()
        self.parent = None
        self._symbol_to_quantity = defaultdict(set)

    def __repr__(self):
        return str(self.uuid)

    def __str__(self):
        QUANTITY_LENGTH_CAP = 35
        building = []
        building += ["Material: " + str(self.uuid), ""]
        for symbol in self._symbol_to_quantity.keys():
            building += ["\t" + symbol.name]
            for quantity in self._symbol_to_quantity[symbol]:
                qs = str(quantity)
                if "\n" in qs or len(qs) > QUANTITY_LENGTH_CAP:
                    qs = "..."
                building += ["\t\t" + qs]
            building += [""]
        return "\n".join(building)

    def add_quantity(self, quantity):
        """
        Adds a property to this material's property graph.  If the
        material has been bound to a Graph instance, correctly adds
        the property to that instance.

        Mutates this graph instance variable and its parent.

        Args:
            quantity (Quantity): property to be bound to the material.

        Returns:
            None
        """
        self._symbol_to_quantity[quantity.symbol].add(quantity)
        if self.parent:
            self.parent._add_quantity(quantity)
        quantity._material.add(self)

    def remove_quantity(self, quantity):
        """
        Removes the Quantity object attached to this Material.

        Args:
            quantity (Quantity): Quantity object reference indicating
            which property is to be removed from this Material.

        Returns:
            None
        """
        if quantity.symbol not in self._symbol_to_quantity:
            raise Exception(
                "Attempting to remove quantity not present in the material.")
        if self.parent:
            self.parent._remove_quantity(quantity)
        self._symbol_to_quantity[quantity.symbol].remove(quantity)
        quantity._material.remove(self)

    def remove_symbol(self, symbol):
        """
        Removes all Quantity Nodes attached to this Material of type symbol.

        Args:
            symbol (Symbol): object indicating which property type
                is to be removed from this material.

        Returns:
            None
        """
        if symbol not in self._symbol_to_quantity:
            raise Exception("Attempting to remove Symbol not present in the material.")
        to_remove = []
        for q in self._symbol_to_quantity[symbol]:
            to_remove.append(q)
        for q in to_remove:
            self.remove_quantity(q)
        del self._symbol_to_quantity[symbol]

    def get_symbols(self):
        """
        Obtains all Symbol objects bound to this Material.

        Returns:
            (set<Symbol>) set containing all symbols bound to this Material.
        """
        return set(self._symbol_to_quantity.keys())

    def get_quantities(self):
        """
        Method obtains all Quantity objects bound to this Material.
        Returns:
            (list<Quantity>) list of all Quantity objects bound to this Material.
        """
        return list(chain.from_iterable(self._symbol_to_quantity.values()))

    def get_unique_quantities(self):
        """
        Returns a set of Quantities that belong ONLY to this material.
        Returns:
            (set<Quantity>)
        """
        to_return = []
        for q_list in self._symbol_to_quantity.values():
            for q in q_list:
                if len(q._material) == 1 and self in q._material:
                    to_return.append(q)
        return to_return

    def get_aggregated_quantities(self):
        """
        Return mean values for all quantities for each symbol.

        Args:
            func (callable): function with which to aggregate quantities

        Returns:
            (dict<Symbol, weighted_mean) mapping from a Symbol to
            an aggregated statistic.
        """
        # TODO: proper weighting system, and more flexibility in object handling
        aggregated = {}
        for symbol, quantities in self._symbol_to_quantity.items():
            if not symbol.category =='object':
                aggregated[symbol] = weighted_mean(list(quantities))
        return aggregated

    @property
    def graph(self):
        """
        Generates a networkX data structure representing the property network and returns
        this object.
        Returns:
            (networkX.multidigraph)
        """
        graph = nx.MultiDiGraph()
        for symbol in self._symbol_to_quantity:
            quantity = self._symbol_to_quantity[symbol]
            graph.add_edge(quantity, symbol)
            graph.add_edge(self, quantity)
        return graph

    def evaluate(self):
        """
        Convenience method to expand this material's properties.
        Creates a Graph instance behind the scenes.
        Mutates this material.
        Returns:
            (None)
        """
        g = Graph()
        g.add_material(self)
        g.evaluate()
