"""
Module containing classes and methods for Material functionality in propnet code.
"""

import networkx as nx

from propnet.core.graph import PropnetNodeType, PropnetNode
from propnet.core.symbols import Symbol

from uuid import uuid4


class Material:
    """
    Class containing methods for creating and interacting with Material objects.

    Under the Propnet infrastructure, Materials are the medium through which properties are communicated. While Model
    and SymbolType nodes create a web of interconnected properties, Materials, as collections of Symbol nodes, provide
    concrete numbers to those properties. At runtime, a Material can be constructed and added to a Propnet instance,
    merging the two graphs and allowing for propagation of concrete numbers through the property web.

    A unique hashcode is stored with each Material upon instantiation. This is used to differentiate between different
    materials at runtime.

    Attributes:
        graph (nx.MultiDiGraph<PropnetNode>): data structure storing all Symbol nodes of the Material.
        id (int): unique hash number used as an identifier for this object.
        root_node (PropnetNode): the Material node associated with this material, has a unique hash id.
        parent (Propnet): Stores a pointer to the Propnet instance this Material has been bound to.
    """
    def __init__(self):
        """
        Creates a Material instance, instantiating a trivial graph of one node.
        """
        self.graph = nx.MultiDiGraph()
        self.id = uuid4()
        self.root_node = PropnetNode(node_type=PropnetNodeType.Material, node_value=self)
        self.graph.add_node(self.root_node)
        self.parent = None

    def add_property(self, property):
        """
        Adds a property to this material's property graph.
        If the material has been bound to a Propnet instance, correctly adds the property to that instance.
        Mutates graph instance variable.

        Args:
            property (Symbol): property to be bound to the material.
        Returns:
            void
        """
        property_node = PropnetNode(node_type=PropnetNodeType.Symbol, node_value=property)
        property_symbol_node = PropnetNode(node_type=PropnetNodeType.SymbolType, node_value=property.type)
        self.graph.add_edge(self.root_node, property_node)
        self.graph.add_edge(property_node, property_symbol_node)
        if self.parent:
            self.parent.graph.add_edge(self.root_node, property_node)
            self.parent.graph.add_edge(property_node, property_symbol_node)

    def available_properties(self):
        """
        Method obtains the names of all properties bound to this Material.

        Returns:
            (list<str>) list of all properties bound to this Material.
        """
        available_propertes = []
        for node in self.graph.nodes:
            if node.node_type == PropnetNodeType.Symbol:
                available_propertes.append(node.value.name)
        return available_propertes
