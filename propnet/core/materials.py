import networkx as nx

from propnet.symbols import SymbolType
from propnet.core.graph import PropnetNodeType
from propnet.core.symbols import Symbol


class Material:

    def __init__(self):

        self.graph = nx.MultiDiGraph()
        self.root_node = PropnetNodeType.Material
        self.graph.add_node(self.root_node)

    def add_property(self, property):
        """

        Args:
          property_type: 
          value: 

        Returns:

        """

        self.graph.add_edge(self.root_node, property)
        self.graph.add_edge(property, property.type)

    def available_properties(self):

        available_propertes = []
        for node in self.graph.nodes:
            if isinstance(node, SymbolType):
                available_propertes.append(node.value.name)

        return available_propertes