from typing import *

import networkx as nx

from propnet import logger
from propnet.models import *
from propnet.symbols import SymbolType, all_symbol_names

from propnet.core.models import AbstractModel
from propnet.core.symbols import Property

from enum import Enum
from collections import Counter, namedtuple

PropnetNodeType = Enum('PropnetNodeType', ['Material', 'SymbolType', 'Symbol', 'Model'])
PropnetNode = namedtuple('PropnetNode', ['node_type', 'node_value'])

class Propnet:
    """ """
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

        # add all symbols to graph
        symbol_nodes = [PropnetNode(node_type=PropnetNodeType.SymbolType, node_value=symbol_type)
                        for symbol_type in SymbolType]
        g.add_nodes_from(symbol_nodes)

        # get a list of our models (except abstract base classes)
        models = [model for model in AbstractModel.__subclasses__()
                  if not model.__module__.startswith('propnet.core')]

        # add all models to graph
        model_nodes = [PropnetNode(node_type=PropnetNodeType.Model, node_value=model)
                       for model in models]
        g.add_nodes_from(model_nodes)

        for model_cls in models:

            model = model_cls()
            model_node = PropnetNode(node_type=PropnetNodeType.Model, node_value=model)

            # there are multiple routes for multiple sets of inputs/outputs (given by idx)
            for idx, connection in enumerate(model.connections):

                outputs, inputs = connection['outputs'], connection['inputs']

                if isinstance(outputs, str):
                    outputs = [outputs]
                if isinstance(inputs, str):
                    inputs = [inputs]

                for input in inputs:
                    symbol_type = SymbolType[model.symbol_mapping[input]]
                    input_node = PropnetNode(node_type=PropnetNodeType.SymbolType,
                                             node_value=symbol_type)
                    g.add_edge(input_node, model_node, route=idx)

                for output in outputs:
                    symbol_type = SymbolType[model.symbol_mapping[output]]
                    output_node = PropnetNode(node_type=PropnetNodeType.SymbolType,
                                              node_value=symbol_type)
                    g.add_edge(model_node, output_node, route=idx)

        self.graph = g

    def add_material(self, material):
        """Add a material and its associated properties to the
        Propnet graph.

        Args:
          material: an instance of a Material

        Returns:

        """

        self.graph = nx.compose(material.graph, self.graph)

    def evaluate(self, material=None, property_type=None):
        """

        Args:
          material:  (Default value = None)
          property_type:  (Default value = None)

        Returns:

        """
        #Get existing Symbol nodes, 'active' SymbolType nodes, and 'candidate' Models.
        symbol_nodes = self.nodes_by_type(PropnetNodeType.Symbol)
        active_symbol_type_nodes = set()
        for node in symbol_nodes:
            active_symbol_type_nodes += \
                list(filter(lambda n: n.node_type == PropnetNodeType.SymbolType, self.graph.neighbors(node)))
        candidate_models = set()
        for node in active_symbol_type_nodes:
            candidate_models += list(filter(lambda n: n.node_type == PropnetNodeType.Model, self.graph.neighbors(node)))

        #Create fast-lookup datastructure (SymbolType -> Symbol):
        lookup_dict = {}
        for node in symbol_nodes:
            if node.node_value.type not in lookup_dict:
                lookup_dict[node.node_value.type] = [node.node_value]
            else:
                lookup_dict[node.node_value.type] += [node.node_value]

        #For each candidate model, check if we have active property types to match types and assumptions.
        #If so, produce the available output properties.
        outputs = []
        for model_node in candidate_models:
            ##Cache necessary data from model_node: input symbols, types, and conditions.
            model = model_node.node_value
            legend = model.symbol_mapping
            sym_inputs = model.input_symbols
            input_conditions = model.conditions
            sym_outputs = model.output_symbols

            def get_types(symbols_in, legend):
                """Converts symbols used in equations to SymbolType enum objects"""
                to_return = []
                for l in symbols_in:
                    out = []
                    for i in l:
                        out.append(SymbolType[legend[symbols_in[l][i]]])
                    to_return.append(out)
                return to_return

            type_inputs = get_types(sym_inputs, legend)
            type_outputs = get_types(sym_outputs, legend)

            ##Look through all input sets and match with all combinations from lookup_dict.
            def gen_input_dicts(symbols, candidate_props, constraint_props, level):
                """Recursively generates all possible combinations of input arguments"""
                current_level = []
                candidates = candidate_props[level]
                for candidate in candidates:
                    if constraint_props[symbols[level]](candidate):
                        current_level.append({symbols[level]: candidate})
                if level == 0:
                    return current_level
                else:
                    others = gen_input_dicts(symbols, candidate_props, level-1)
                    to_return = []
                    for entry1 in current_level:
                        for entry2 in others:
                            to_return.append(entry1 + entry2)
                    return to_return

            for i in range(0, len(type_inputs)):
                candidate_properties = []
                for j in range(0, len(type_inputs[i])):
                    candidate_properties.append(lookup_dict.get(type_inputs[i][j], []))
                input_sets = gen_input_dicts(sym_inputs, candidate_properties, input_conditions, len(candidate_properties)-1)
                for input_set in input_sets:
                    outputs.append(model.evaluate(input_set))
        return outputs

    def shortest_path(self, property_one: str, property_two: str):
        """

        Args:
          property_one: str: 
          property_two: str: 

        Returns:

        """
        # very easy to do with networkx, use in-built algo
        return NotImplementedError

    def populate_with_test_values(self):
        """ """
        # takes test values from the property definitions
        return NotImplementedError

    def nodes_by_type(self, node_type):
        """:return: Return a list of nodes of property types."""
        return list(filter(lambda n: n.node_type == node_type, self.graph.nodes))

    @property
    def all_models(self):
        """:return: Return a list of nodes of models."""
        return list(filter(lambda x: issubclass(x, AbstractModel), self.graph.nodes))

    @property
    def model_tags(self):
        """Collates tags present in models.
        :return: Returns list of tags

        Args:

        Returns:

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
