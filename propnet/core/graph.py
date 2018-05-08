"""
Module containing classes and methods for graph functionality in Propnet code.
"""

from typing import *

import networkx as nx

from propnet import logger
from propnet.models import DEFAULT_MODELS
from propnet.symbols import DEFAULT_SYMBOL_TYPES

from propnet.core.quantity import Quantity
from propnet.core.models import AbstractModel

from enum import Enum
from collections import Counter, namedtuple
from uuid import UUID, uuid4


_ALLOWED_NODE_TYPES = ('Material', 'Symbol', 'Quantity', 'Model')


# TODO: this should be replaced with a proper class
PropnetNodeType = Enum('PropnetNodeType', _ALLOWED_NODE_TYPES)
PropnetNode = namedtuple('PropnetNode', ['node_type', 'node_value'])
PropnetNode.__repr__ = lambda self: "{}<{}>".format(self.node_type.name,
                                                    self.node_value.__repr__())


class Propnet:
    """
    Class containing methods for creating and interacting with a Property Network.

    The Property Network contains a set of PropnetNode namedtuples with connections stored as directed edges between
    the nodes.

    Upon initialization a base graph is constructed consisting of all valid SymbolTypes and Models found in surrounding
    folders. These are Symbol and Model node_types respectively. Connections are formed between the nodes based on
    given inputs and outputs of the models. At this stage the graph represents a symbolic web of properties without
    any actual input values.

    Materials and Properties / Conditions can be added at runtime using appropriate support methods. These methods
    dynamically create additional PropnetNodes and edges on the graph of Material and Quantity node_types respectively.

    Given a set of Materials and Properties / Conditions, the symbolic web of properties can be utilized to predict
    values of connected properties on demand.

    Attributes:
        graph (nx.MultiDiGraph<PropnetNode>): data structure supporting the property network.

    """

    def __init__(self, materials=None, models=None, symbol_types=None):
        """
        Creates a Propnet instance
        """

        # set our defaults if no models/symbol types supplied
        models = models or DEFAULT_MODELS
        symbol_types = symbol_types or DEFAULT_SYMBOL_TYPES

        # create the graph
        self.graph = nx.MultiDiGraph()

        # add our symbols
        self._symbol_types = symbol_types
        self.add_symbol_types(symbol_types)

        # add our models
        self.add_models(models)

        # add appropriate edges to the graph
        for model in models.values():

            model = model(symbol_types=self._symbol_types)  # instantiate model
            model_node = PropnetNode(node_type=PropnetNodeType.Model, node_value=model)

            # integer idx is used to disambiguate edges when
            # multiple paths exist between the same start and end nodes
            for idx, connection in enumerate(model.connections):

                outputs, inputs = connection['outputs'], connection['inputs']

                if isinstance(outputs, str):
                    outputs = [outputs]
                if isinstance(inputs, str):
                    inputs = [inputs]

                for input in inputs:
                    symbol_type = symbol_types[model.symbol_mapping[input]]
                    input_node = PropnetNode(node_type=PropnetNodeType.Symbol,
                                             node_value=symbol_type)
                    self.graph.add_edge(input_node, model_node, route=idx)

                for output in outputs:
                    symbol_type = symbol_types[model.symbol_mapping[output]]
                    output_node = PropnetNode(node_type=PropnetNodeType.Symbol,
                                              node_value=symbol_type)
                    self.graph.add_edge(model_node, output_node, route=idx)

        if materials:
            for material in materials:
                self.add_material(material)

    def add_models(self, models):
        """
        Add a user-defined model to the Propnet graph.

        Args:
            models: An instance of a model class (subclasses AbstractModel)

        Returns:

        """
        model_nodes = [PropnetNode(node_type=PropnetNodeType.Model,
                                   node_value=model(symbol_types=self._symbol_types))
                       for model in models.values()]
        self.graph.add_nodes_from(model_nodes)

    def add_symbol_types(self, symbol_types):
        """

        Args:
            symbol_types: {name:Symbol}

        Returns:

        """
        self._symbol_types.update(symbol_types)
        symbol_type_nodes = [PropnetNode(node_type=PropnetNodeType.Symbol,
                                         node_value=symbol_type)
                             for symbol_type in symbol_types.values()]

        self.graph.add_nodes_from(symbol_type_nodes)

    def nodes_by_type(self, node_type):
        """
        Gathers all PropnetNodes of a given PropnetNodeType.

        Args:
            node_type (str): type of node that will be returned.
        Returns:
            (list<PropnetNode>) list of nodes of property types.
        """
        if node_type not in _ALLOWED_NODE_TYPES:
            raise ValueError("Unsupported node type, choose from: {}"
                             .format(_ALLOWED_NODE_TYPES))
        return filter(lambda n: n.node_type.name == node_type, self.graph.nodes)

    def add_material(self, material):
        """
        Add a material and any of its associated properties to the Propnet graph.
        Mutates the graph instance variable.

        Args:
            material (Material) Material whose information will be added to the graph.
        Returns:
            void
        """
        material.parent = self
        self.graph = nx.compose(material.graph, self.graph)

    def remove_material(self, material):
        """
        Removes a material and any of its associated properties from the Propnet graph.
        Mutates the graph instance variable.

        Args:
            material (Material) Material whose information will be removed from the graph.
        Returns:
            void
        """
        symbol_nodes = self.graph.neighbors(material.root_node)
        self.graph.remove_node(material.root_node)
        for symbol_node in symbol_nodes:
            if any([x.node_type == 'Material' for x in self.graph.neighbors(symbol_node)]):
                continue
            self.graph.remove_node(symbol_node)

    def evaluate(self, material=None, property_type=None):
        """
        Expands the graph, producing the output of models that have the appropriate inputs supplied.
        Mutates the graph instance variable.

        Optional arguments limit the scope of which models or properties are tested.
            material parameter: produces output from models only if the input properties come from the specified material.
                                mutated graph will modify the Material's graph instance as well as this graph instance.
                                mutated graph will include edges from Material to Quantity to Symbol.
            property_type parameter: produces output from models only if the input properties are in the list.

        If no material parameter is specified, the generated SymbolNodes will be added with edges to and from
        corresponding SymbolTypeNodes specifically. No connections will be made to existing Material nodes because
        a Quantity might be derived from a combination of materials in this case. Likewise existing Material nodes' graph
        instances will not be mutated in this case.

        Args:
            material (Material): optional limit on which material's properties will be expanded (default: all materials)
            property_type (list<Symbol>): optional limit on which Symbols will be considered as input.
        Returns:
            void
        """

        ##
        # Get existing Quantity nodes, 'active' Symbol nodes, and 'candidate' Models.
        # Filter by provided material and property_type arguments.
        ##

        if not material:
            # All symbol_nodes are candidates for evaluation.
            symbol_nodes = list(self.nodes_by_type('Quantity'))
        else:
            # Only symbol_nodes connected to the given Material object are candidates for evaluation.
            material_nodes = self.nodes_by_type('Material')
            material_node = None
            for node in material_nodes:
                if node.node_value == material:
                    if material_node:
                        raise ValueError('Multiple identical materials found.')
                    material_node = node
            if not material_node:
                raise ValueError('Specified material not found.')
            symbol_nodes = []
            for node in self.graph.neighbors(material_node):
                if node.node_type == PropnetNodeType['Quantity'] and node not in symbol_nodes:
                    symbol_nodes.append(node)

        if property_type:
            # Only Symbol objects in the property_type list are candidates for evaluation.
            symbol_nodes = [node for node in symbol_nodes
                            if node.node_value.symbol in property_type]

        # Get set of SymbolTypes that have values provided.
        active_symbol_type_nodes = set()
        for node in symbol_nodes:
            for neighbor in self.graph.neighbors(node):
                if neighbor.node_type != PropnetNodeType.Symbol:
                    continue
                active_symbol_type_nodes.add(neighbor)

        # Get set of Models that have values provided to inputs.
        candidate_models = set()
        for node in active_symbol_type_nodes:
            for neighbor in self.graph.neighbors(node):
                if neighbor.node_type != PropnetNodeType.Model:
                    continue
                candidate_models.add(neighbor.node_value)

        ##
        # Define helper data structures and methods.
        ##

        # Create fast-lookup data structure Dict[Symbol, [Quantity]]:
        lookup_dict = {}
        for node in symbol_nodes:
            if node.node_value.symbol not in lookup_dict:
                lookup_dict[node.node_value.symbol] = [node.node_value]
            else:
                lookup_dict[node.node_value.symbol] += [node.node_value]

        # Create fast-lookup data structure Dict[Quantity, MaterialNode]
        symbol_to_material_dict = {}

        # Create fast-lookup data structure (str (Symbol name) -> Symbol)
        symbol_type_nodes = self.nodes_by_type('Symbol')
        symbol_types = {x.node_value.name: x.node_value for x in symbol_type_nodes}

        def get_source_nodes(graph, node):
            # TODO: store material uuid(s) in symbol (?)
            """
            Given a Quantity node on the graph, returns a list of connected material nodes.
            This list symbolizes the set of materials for which this Quantity is a property.
            Args:
                graph (networkx.MultiDiGraph): graph on which the node is stored.
                node (PropnetNode): node on the graph whose connected material nodes are to be found.
            Returns:
                (List[PropnetNode]): list of material type nodes that are connected to this node.
            """
            to_return = []
            for n in graph.in_edges(node):
                if n[0].node_type == PropnetNodeType['Material']:
                    to_return.append(n[0])
            return to_return

        for node in symbol_nodes:
            symbol_to_material_dict[node.node_value] = get_source_nodes(self.graph, node)

        ##
        # For each candidate model, check if we have active property types to match inputs and conditions.
        # If so, produce the available output properties using all possible permutations of inputs & add
        #     new models that can be calculated from previously-derived properties.
        # If certain outputs have already been generated, do not generate duplicate outputs.
        ##

        # Keeps track of number of SymbolTypes derived from the current loop iteration.
        # Loop terminates when no new properties are derived from any models.

        original_models = {x for x in candidate_models}
        evaluated_models = set()
        next_round_models = candidate_models

        added_on_loop = True

        while added_on_loop:
            added_on_loop = False
            candidate_models = next_round_models
            next_round_models = set()

            for model in candidate_models:

                # TODO: move get_types to model

                outputs = []
                # Cache necessary data from model_node: input symbols, types, and conditions.
                legend = model.symbol_mapping
                sym_inputs = model.input_symbols
                for i in sym_inputs:
                    for c in model.constraint_symbols:
                        if c not in i:
                            i.append(c)

                def get_types(symbols_in, legend, symbol_types):
                    """Converts symbols used in equations to Symbol objects"""
                    # TODO: move to model
                    to_return = []
                    for l in symbols_in:
                        if not isinstance(l, list):
                            l = [l]
                        out = []
                        for i in l:
                            to_append = symbol_types.get(legend[i])
                            if not to_append:
                                raise Exception('Error evaluating graph: Model references Symbol'
                                                'objects that do not appear in the graph.')
                            out.append(to_append)
                        to_return.append(out)
                    return to_return

                # list<list<Symbol>>, representing sets of input properties the model accepts.
                type_inputs = get_types(sym_inputs, legend, symbol_types)

                # Recursive helper method.
                # Look through all input sets and match with all combinations from lookup_dict.
                # TODO: look at itertools.permutations
                def gen_input_dicts(symbols, candidate_props, level):
                    """
                    Recursively generates all possible combinations of input arguments.
                    Args:
                        symbols (list<str>):
                            one set of input symbols required by the model.
                        candidate_props (list<list<Quantity>>):
                            list of potential values that can be plugged into each symbol,
                            the outer list corresponds by ordering to the symbols list,
                            the inner list gives values that can be plugged in to each symbol.
                        level (int):
                            internal parameter used for recursion, says which symbol is being enumerated, should
                                     be set to the final index value of symbols.
                    Returns:
                        (list<dict<String, Quantity>>) list of dictionaries giving symbol strings mapped to values.
                    """
                    current_level = []
                    candidates = candidate_props[level]
                    for candidate in candidates:
                        current_level.append({symbols[level]: candidate})
                    if level == 0:
                        return current_level
                    else:
                        others = gen_input_dicts(symbols, candidate_props, level-1)
                        to_return = []
                        for entry1 in current_level:
                            for entry2 in others:
                                merged_dict = {}
                                for (k, v) in entry1.items():
                                    merged_dict[k] = v
                                for (k, v) in entry2.items():
                                    merged_dict[k] = v
                                to_return.append(merged_dict)
                        return to_return

                # Get candidate input Symbols for the given model.
                # Skip over any input Quantity lists that have already been evaluated.
                for i in range(0, len(type_inputs)):
                    candidate_properties = []
                    for j in range(0, len(type_inputs[i])):
                        candidate_properties.append(lookup_dict.get(type_inputs[i][j], []))
                    input_sets = gen_input_dicts(sym_inputs[i], candidate_properties,
                                                 len(candidate_properties)-1)
                    for input_set in input_sets:
                        if not model.check_constraints(input_set):
                            continue
                        plug_in_set = {}
                        sourcing = set()
                        for (k, v) in input_set.items():
                            plug_in_set[k] = v.value
                            for elem in symbol_to_material_dict[v]:
                                sourcing.add(elem)
                        outputs.append({"output": model.evaluate(plug_in_set), "source": sourcing})

                # For any new outputs generated, create the appropriate SymbolNode and connections to SymbolTypeNodes
                # For any new outputs generated, create the appropriate connections from Material Nodes
                # For any new outputs generated, add new models connected to the derived SymbolTypeNodes to the
                #     candidate_models list & update convenience data structures.
                # Mutates this graph.
                symbol_outputs = []
                output_sources = []
                if len(outputs) == 0:
                    next_round_models.add(model)
                else:
                    added_on_loop = True
                    evaluated_models.add(model)
                for entry in outputs:
                    for (k, v) in entry['output'].items():
                        prop_type = symbol_types.get(legend.get(k))
                        if not prop_type:
                            continue
                        symbol_outputs.append(Quantity(prop_type, v, None))
                        output_sources.append(entry['source'])
                for i in range(0, len(symbol_outputs)):
                    # Add outputs to graph.
                    symbol = symbol_outputs[i]
                    symbol_node = PropnetNode(node_type=PropnetNodeType['Quantity'], node_value=symbol)
                    if symbol_node in self.graph:
                        continue
                    symbol_type_node = PropnetNode(node_type=PropnetNodeType['Symbol'], node_value=symbol.symbol)
                    self.graph.add_edge(symbol_node, symbol_type_node)
                    for source_node in output_sources[i]:
                        self.graph.add_edge(source_node, symbol_node)

                    # Strategy A:

                    if len(output_sources[i]) == 1:
                        store = output_sources[i].__iter__().__next__()
                        store.node_value.graph.add_edge(store.node_value.root_node, symbol_node)

                    # Strategy B:
                    """
                    for store in output_sources[i]:
                        store.node_value.graph.add_edge(store.node_value.root_node, symbol_node)
                    """

                    # Update helper data structures etc. for next cycle.
                    symbol_to_material_dict[symbol] = get_source_nodes(self.graph, symbol_node)
                    if symbol.symbol not in lookup_dict:
                        lookup_dict[symbol.symbol] = [symbol]
                    else:
                        lookup_dict[symbol.symbol] += [symbol]
                    if not property_type or symbol.symbol in property_type:
                        for neighbor in self.graph.neighbors(symbol_type_node):
                            if neighbor.node_type == PropnetNodeType['Model']:
                                if neighbor.node_value not in original_models:
                                    next_round_models.add(neighbor.node_value)

    def shortest_path(self, property_one: str, property_two: str):
        """ """
        # very easy to do with networkx, use in-built algo
        return NotImplementedError

    def populate_with_test_values(self):
        """ """
        # takes test values from the property definitions
        return NotImplementedError


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

    def __repr__(self):
        nodes = "\n".join([n.__repr__() for n in self.graph.nodes()])
        edges = "\n".join(["\t{}-->{}".format(u.__repr__(), v.__repr__())
                           for u, v in self.graph.edges()])
        return "Nodes:\n{}\nEdges:\n{}".format(nodes, edges)

    def __str__(self):
        """
        Returns a full summary of the graph in terms of the SymbolTypes, Symbols, Materials, and Models
        that it contains. Connections are shown as nesting within the printout.

        Returns:
            (str) representation of this Propnet object.
        """
        summary = ["Propnet Graph", ""]
        property_type_nodes = self.nodes_by_type('Symbol')
        summary += ["Quantity Types:"]
        for property_type_node in property_type_nodes:
            summary += ["\t " + property_type_node.node_value.display_names[0]]
            neighbors = self.graph.neighbors(property_type_node)
            for neighbor in neighbors:
                if neighbor.node_type != PropnetNodeType['Model']:
                    continue
                summary += ["\t\t " + neighbor.node_value.title]
        model_nodes = list(self.nodes_by_type('Model'))
        summary += ["Models:"]
        for model_node in model_nodes:
            summary += ["\t " + model_node.node_value.title]
            neighbors = self.graph.neighbors(model_node)
            for neighbor in neighbors:
                if neighbor.node_type != PropnetNodeType['Symbol']:
                    continue
                summary += ["\t\t " + neighbor.node_value.display_names[0]]
        materials = list(self.nodes_by_type('Material'))
        if len(materials) != 0:
            summary += ["Materials:"]
        for material_node in materials:
            summary += ["\t " + str(material_node.node_value.uuid)]
            for property in filter(lambda n: n.node_type == PropnetNodeType['Quantity'],
                                   self.graph.neighbors(material_node)):
                summary += ["\t\t " + property.node_value.symbol.display_names[0] +
                            "\t:\t" + str(property.node_value)]
        return "\n".join(summary)
