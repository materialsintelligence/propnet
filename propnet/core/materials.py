import networkx as nx

from propnet.properties import PropertyType
from propnet.core.graph import NodeType
from propnet.core.properties import Property

class Material:

    def __init__(self):

        self.graph = nx.MultiDiGraph()
        self.root_node = NodeType.Material
        self.graph.add_node(self.root_node)

    @classmethod
    def material_from_mpid(cls, mpid):
        return NotImplementedError

    def add_property(self, property_type, value):

        property = Property(property_type, value)

        self.graph.add_edge(self.root_node, property)
        self.graph.add_edge(property, PropertyType[property_type])


    def add_formula(self, formula: str, x: float = 1.0):
        """

        :param formula:
        :param x: proportion of material, e.g. 1.0
        for a single-phase material
        :return:
        """
        self.check_total(x)
        self.graph.add_edge(self.root_node,
                            formula, x=x)

    def add_pymatgen_object(self, object, x=1.0):
        # this may change
        self.check_total(x)
        self.graph.add_edge(self.root_node,
                            object.as_dict(), x=x)

    def add_structure(self, structure, x=1.0):
        return self.add_pymatgen_object(structure, x=x)

    def check_total(self, new_x):
        x_total = new_x
        for u, v, data in self.graph.edges(data=True):
            x_total += data.get('x', 0)
        if x_total > 1.0:
            raise ValueError("Too many formulae/structures have "
                             "been defined for this material!")
