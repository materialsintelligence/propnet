from typing import *

import networkx as nx

from propnet import logger
from propnet.models import *
from propnet.properties import PropertyType, all_property_names

from propnet.core.properties import Property
from propnet.core.models import AbstractModel

from enum import Enum

# TODO: add more node types as appropriate
NodeType = Enum('NodeType', ['Material'])

class Propnet:

    # clumsy implementation at present, storing classes directly
    # on graph ... this may be fine, but more investigation required!

    def __init__(self):
        """
        Create a Propnet instance. This will contain all models
        and all property types. A Material graph, with material-specific
        properties, can be composed into it to solve for new material
        properties.
        """

        g = nx.MultiDiGraph()

        # add all our property types
        g.add_nodes_from(PropertyType)

        # add all our models
        # filter out abstract base classes
        models = [model for model in AbstractModel.__subclasses__()
                  if not model.__module__.startswith('propnet.core')]
        g.add_nodes_from(models)

        for model_cls in models:
            model = model_cls()
            for idx, (output, inputs) in enumerate(model.connections.items()):

                for input in inputs:
                    input = PropertyType[model.symbol_mapping[input]]
                    g.add_edge(input, model_cls, route=idx)

                output = PropertyType[model.symbol_mapping[output]]

                # there are multiple routes for multiple sets of inputs/outputs
                g.add_edge(model_cls, output, route=idx)

        self.graph = g

    def add_material(self, material):
        """
        Add a material and its associated properties to the
        Propnet graph.

        :param material: an instance of a Material
        :return:
        """

        self.graph = nx.compose(material.graph, self.graph)

    def evaluate(self, material=None, property_type=None):
        # return as pandas data frame
        return NotImplementedError

    def populate_with_test_values(self):
        # takes test values from the property definitions
        return NotImplementedError

    @property
    def property_type_nodes(self):
        return list(filter(lambda x: isinstance(x, PropertyType), self.graph.nodes))

    @property
    def model_nodes(self):
        return list(filter(lambda x: issubclass(x, AbstractModel), self.graph.nodes))

    def __str__(self):
        summary = ["Propnet Graph", "", "Property Types:"]
        summary += ["\t "+n.value.display_names[0] for n in self.property_type_nodes]
        summary += ["Models:"]
        summary += ["\t "+n().title for n in self.model_nodes]
        return "\n".join(summary)