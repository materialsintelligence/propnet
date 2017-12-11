from typing import *

import networkx as nx

from propnet import logger
from propnet.models import *
from propnet.properties import PropertyType, all_property_names

from propnet.core.properties import Property
from propnet.core.models import AbstractModel
from propnet.core.models import AbstractAnalyticalModel

from enum import Enum
from collections import Counter

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

        # add all our models (except abstract base classes)
        abstract_models = [model for model in AbstractModel.__subclasses__()
                                     if not model.__module__.startswith('propnet.core')]
        abstract_analytical_models = [model for model in AbstractAnalyticalModel.__subclasses__()
                                                if not model.__module__.startswith('propnet.core')]
        models = abstract_models + abstract_analytical_models
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
        # should be straight-forward to evaluate graph,
        # filter graph on what material properties have been
        # defined, create a sub-graph, and traverse this sub-graph
        # all model edges are directed for inputs and outputs, so
        # traversal shouldn't require too much logic
        # return as pandas data frame
        return NotImplementedError

    def shortest_path(self, property_one: str, property_two: str):
        # very easy to do with networkx, use in-built algo
        return NotImplementedError

    def populate_with_test_values(self):
        # takes test values from the property definitions
        return NotImplementedError

    @property
    def property_type_nodes(self):
        """
        :return: Return a list of nodes of property types.
        """
        return list(filter(lambda x: isinstance(x, PropertyType), self.graph.nodes))

    @property
    def all_models(self):
        """
        :return: Return a list of nodes of models.
        """
        return list(filter(lambda x: issubclass(x, AbstractModel), self.graph.nodes))

    @property
    def model_tags(self):
        """
        Collates tags present in models.
        :return: Returns list of tags
        """
        all_tags = [tag for tags in self.all_models for tag in tags]
        unique_tags = sorted(list(set(all_tags)))
        return Counter(all_tags)

    def __str__(self):
        """
        :return:  Return a summary of the graph. Doesn't show any defined materials yet.
        """
        summary = ["Propnet Graph", "", "Property Types:"]
        summary += ["\t " + n.value.display_names[0] for n in self.property_type_nodes]
        summary += ["Models:"]
        summary += ["\t " + n().title for n in self.all_models]
        return "\n".join(summary)
