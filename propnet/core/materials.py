"""
Module containing classes and methods for Material functionality in propnet code.
"""

import networkx as nx

from collections import defaultdict
from typing import *

from propnet.core.graph import NodeType, Node
from propnet.core.symbols import Symbol
from propnet.core.quantity import Quantity, weighted_mean
from propnet.core.utils import uuid

class Material:
    """
    Class containing methods for creating and interacting with Material objects.

    Under the Propnet infrastructure, Materials are the medium through which properties are
    communicated. While Model
    and Symbol nodes create a web of interconnected properties, Materials, as collections of
    Quantity nodes, provide
    concrete numbers to those properties. At runtime, a Material can be constructed and added to
    a Propnet instance,
    merging the two graphs and allowing for propagation of concrete numbers through the property
    web.

    A unique hashcode is stored with each Material upon instantiation. This is used to
    differentiate between different
    materials at runtime.

    Attributes:
        graph (nx.MultiDiGraph<Node>): data structure storing all Quantity nodes of the
        Material.
        uuid (int): unique hash number used as an identifier for this object.
        root_node (Node): the Material node associated with this material, has a unique
        hash id.
        parent (Propnet): Stores a pointer to the Propnet instance this Material has been bound to.
    """
    def __init__(self):
        """
        Creates a Material instance, instantiating a trivial graph of one node.
        """
        self.uuid = uuid()

        self.graph = nx.MultiDiGraph()
        self.root_node = Node(node_type=NodeType.Material, node_value=self)
        self.graph.add_node(self.root_node)

        self.parent = None

    @property
    def subgraph(self):
        """
        Returns just the root material node + its associated symbols.
        """
        material_nodes = [self.root_node] \
                         + list(self.graph.predecessors(self.root_node))\
                         + list(self.graph.successors(self.root_node))
        return self.graph.subgraph(material_nodes)

    def add_quantity(self, property):
        """
        Adds a property to this material's property graph.
        If the material has been bound to a Propnet instance, correctly adds the property to that
        instance.
        Mutates graph instance variable.

        Args:
            property (Quantity): property to be bound to the material.
        Returns:
            void
        """
        property_node = Node(node_type=NodeType.Quantity, node_value=property)
        property_symbol_node = Node(node_type=NodeType.Symbol,
                                    node_value=property.symbol)
        self.graph.add_edge(self.root_node, property_node)
        self.graph.add_edge(property_node, property_symbol_node)
        if self.parent:
            self.parent.graph.add_edge(self.root_node, property_node)
            self.parent.graph.add_edge(property_node, property_symbol_node)

    def remove_quantity(self, property):
        """
        Removes the Quantity object attached to this Material.
        Args:
            property (Quantity): Quantity object reference indicating with property is to be
            removed from this Material.
        Returns:
            None
        """
        nodes_to_remove = []
        for node in self.graph.neighbors(self.root_node):
            if node.node_value == property:
                nodes_to_remove.append(node)
                if self.parent:
                    self.parent.graph.remove_node(node)
        for node in nodes_to_remove:
            self.graph.remove_node(node)

    def remove_property_type(self, property_type):
        """
        Removes all Quantity Nodes attached to this Material whose Symbol matches the indicated
        property_type text.
        Args:
            property_type (str): String indicating which property type is to be removed from this material.
        Returns:
            None
        """
        for node in self.graph.neighbors(self.root_node):
            if node.node_value.symbol.name == property_type:
                self.graph.remove_node(node)
                if self.parent:
                    self.parent.graph.remove_node(node)

    def available_properties(self):
        """
        Method obtains the names of all properties bound to this Material.

        Returns:
            (list<str>) list of all properties bound to this Material.
        """
        available_propertes = []
        for node in self.graph.nodes:
            if node.node_type == NodeType.Quantity:
                available_propertes.append(node.node_value.symbol.name)
        return available_propertes

    def available_quantity_nodes(self):
        """
        Method obtains all Quantity objects bound to this Material.

        Returns:
            (list<Node<Quantity>>) list of all Quantity objects bound to this Material.
        """
        to_return = []
        for node in self.graph.nodes:
            if node.node_type == NodeType['Quantity']:
                to_return.append(node)
        return to_return

    def __repr__(self):
        return str(self.uuid)

    def __str__(self):
        to_return = "Material: " + str(self.uuid) + "\n"
        for node in self.available_quantity_nodes():
            to_return += "\t" + node.node_value.symbol.name + ":\t"
            to_return += str(node.node_value.value) + "\n"
        return to_return

    @property
    def quantities_grouped_by_symbol(self) ->  Dict[Symbol, List[Quantity]]:
        grouped_quantities = defaultdict(list)
        for node in self.available_quantity_nodes():
            symbol = node.node_value.symbol
            grouped_quantities[symbol].append(node.node_value)
        return grouped_quantities

    def aggregate(self):

        new_qs = {}
        for symbol, quantities in self.quantities_grouped_by_symbol.items():

            if len(quantities) > 1:

                new_quantity = weighted_mean(quantities)

                if new_quantity:
                    new_qs[symbol.name] = new_quantity
                    for quantity in quantities:
                        self.remove_quantity(quantity)
                    self.add_quantity(new_quantity)

        # fake for demo
        return new_qs