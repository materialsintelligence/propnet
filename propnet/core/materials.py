import networkx as nx


from propnet.symbols import SymbolType
from propnet.core.graph import PropnetNodeType
from propnet.core.symbols import Symbol

from uuid import uuid4


class Material:

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.root_node = Node(node_type=NodeType.Material, node_value=str(uuid4()))
        self.graph.add_node(self.root_node)

    def add_property(self, property):
        """

        Args:
          property_type: 
          value: 

        Returns:

        """
        property_node = Node(node_type=NodeType.Symbol, node_value=property)
        property_symbol_node = Node(node_type=NodeType.SymbolType, node_value=property.type)
        self.graph.add_edge(self.root_node, property_node)
        self.graph.add_edge(property_node, property_symbol_node)

    def available_properties(self):
        available_propertes = []
        for node in self.graph.nodes:
            if node.node_type == NodeType.Symbol:
                available_propertes.append(node.value.name)
        return available_propertes