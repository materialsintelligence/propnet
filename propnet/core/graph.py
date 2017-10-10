import networkx as nx

from propnet.core.properties import Property, PropertyQuantity
from typing import NamedTuple, Any
from enum import Enum

NodeType = Enum('NodeType', ['Material',
                             'PropertyType',
                             'PropertyQuantity',
                             'PymatgenObject',
                             'Generic',
                             'Tag',
                             'Model'])

Node = NamedTuple('Node', [('type', NodeType)
                           ('value', Any)])


class Propnet:

    def __init__(self):

        pass

    def evaluate(self, material=None, property_type=None):
        pass

    def to_json(self):
        pass

PROPNET = Propnet()

class Material:

    def __init__(self,
                 formula=None,
                 structure=None,
                 tags=None,
                 propnet=None):

        self.propnet = propnet if propnet else PROPNET

        self.root = Node(type=NodeType.Material,
                        value=None)

        self.propnet.add_node(self.root)

    def add_formula(self, formula, weight=1.0):

        self.formula = Node(type=NodeType.Generic,
                       value=formula)

        self.propnet.add_edge(self.root_node,
                              self.formula,
                              weight=weight)

    def add_structure(self, structure, weight=1.0):
        self.add_pymatgen_object(structure, weight=weight)

    def add_pymatgen_object(self, object, weight=1.0):

        structure = Node(type=NodeType.PymatgenObject,
                         value=object)

        self.propnet.add_edge(self.root_node,
                              structure,
                              weight=weight)

    def add_property(self, property_type, quantity):
        pass

    def retrieve_property(self, property_type):
        pass