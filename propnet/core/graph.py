"""Knowledge graph builder and traversal.

This module contains the class responsible for building the propnet knowledge graph and
traversing it when materials properties are applied to it in order to derive all possible
new materials properties.

The knowledge graph is built from the contents of the propnet model and symbol ``Registry``.
In order to use models and symbols that are built-in to propnet, be sure to:

>>> import propnet.models    # For symbols and models
>>> import propnet.symbols   # For symbols only

Examples:
    To run the knowledge graph on a material, import the models and symbols desired from
    the built-in library (see above) or add your own custom models/symbols to the ``Registry``:

    >>> from propnet.core.symbols import Symbol
    >>> from propnet.core.models import EquationModel    # Or whichever type you are making
    >>> from propnet.core.materials import Material
    >>> sym1 = Symbol(...)      # Register custom symbols (register=True is default)
    >>> sym2 = Symbol(...)
    >>> model1 = EquationModel(...)     # Register custom model (register=True is default)

    Then you can run the graph traversal algorithm using the models chosen.

    >>> from propnet.core.graph import Graph
    >>> g = Graph(parallel=True)    # Can run in parallel mode
    >>> material = Material(...)    # Create a material with some properties
    >>> evaluated_material = g.evaluate(material)   # Run graph traversal on material
"""

import logging
from collections import defaultdict
from itertools import product, chain, repeat
from collections import deque
import concurrent.futures
from functools import partial
from multiprocessing import cpu_count
import copy
from typing import Set, Dict, Union

from chronic import Timer, timings, clear as clear_timings
from pandas import DataFrame
import numpy as np
import networkx as nx

from propnet.core.materials import CompositeMaterial
from propnet.core.materials import Material
from propnet.core.models import Model, CompositeModel
from propnet.core.quantity import QuantityFactory
from propnet.core.provenance import SymbolTree, TreeElement
from propnet.core.symbols import Symbol
from propnet.core.utils import Timeout
from propnet.core.registry import Registry

logger = logging.getLogger(__name__)
"""logging.Logger: Logger for debugging"""


class Graph:
    """
    Class containing methods for creating and interacting with the propnet knowledge graph.
    This graph accepts a material (or multiple for composite models) and recursively applies
    the knowledge graph to derive all possible property values (quantities) available on
    the graph.

    Notes:
        To use the built-in propnet models, you must explicitly import them using:

        >>> import propnet.models

        This will register the models in the ``Registry`` and allow you to load them.
        If you do not import them or supply them yourself at instantiation, the knowledge
        graph will contain no models.

    Examples:
        The ``evaluate()`` method will be the main entry point for the graph evaluation.

        >>> from propnet.core.graph import Graph
        >>> from propnet.core.materials import Material
        >>> g = Graph()
        >>> material = Material(...)
        >>> evaluated_material = g.evaluate(material)

        However, propnet also has limited support for composite materials (materials made up
        of more than one material). For those materials, use ``evaluate_composite()``.

        >>> from propnet.core.graph import Graph
        >>> from propnet.core.materials import Material, CompositeMaterial
        >>> g = Graph()
        >>> m1 = Material(...)
        >>> m2 = Material(...)
        >>> evaluated_material = g.evaluate_composite(CompositeMaterial([m1, m2]))

        The composite evaluation algorithm is somewhat slow and there are not a large number
        of composite models available in the built-in library. However, we are always
        accepting contributions!

    """

    def __init__(self,
                 models: Dict[str, Model] = None,
                 composite_models: Dict[str, CompositeModel] = None,
                 symbol_types: Dict[str, Symbol] = None,
                 parallel: bool = False,
                 max_workers: int = None) -> None:
        """
        Args:
            models (`dict` or `None`): optional, dict of models to use for graph evaluation,
                keyed by model name. Default: ``None`` (dictionary returned by
                ``Registry('models')``)
            composite_models (`dict` or `None`): optional, dict of composite models to use for
                graph evaluation, keyed by model name. Default: ``None`` (dictionary returned by
                ``Registry('composite_models')``)
            symbol_types (`dict` or `None`): optional, dict of symbols to use for graph evaluation.
                Note all symbols used by the desired models must be included or an error will occur.
                Default: ``None`` (dictionary returned by ``Registry('symbols')``)
            parallel (bool): ``True`` creates a pool of workers for parallel graph evaluation.
                Default: ``False`` (runs serially)
            max_workers (int): Number of workers for parallel worker pool.
                Default: ``None`` (1 for serial, max number of available CPUs for parallel)
        """

        # set our defaults if no models/symbol types supplied
        symbol_types = symbol_types or Registry("symbols")

        # create the graph
        self._symbol_types = dict()
        self._models = dict()
        self._composite_models = dict()
        self._input_to_model = defaultdict(set)
        self._output_to_model = defaultdict(set)
        self._graph_timings = None
        self._model_timings = None

        if parallel:
            self._parallel = True
            if max_workers is None or max_workers > cpu_count():
                self._max_workers = cpu_count()
            else:
                self._max_workers = max_workers
            self._executor = None
        else:
            self._parallel = False
            if max_workers is not None:
                raise ValueError('Cannot specify max_workers with parallel=False')
            self._executor = None
            self._max_workers = 0

        if symbol_types:
            self.update_symbol_types(symbol_types)

        if models is None:
            self.update_models(Registry("models"))
        else:
            self.update_models(models)

        if composite_models is None:
            self.update_composite_models(Registry("composite_models"))
        else:
            self.update_composite_models(composite_models)

    def __str__(self):
        """
        Returns a full summary of the graph in terms of the
        Symbols, Materials, and Models that it contains. Connections are
        shown as nesting within the printout.

        Returns:
            str: representation of this Graph object.
        """
        summary = ["Propnet Printout", ""]
        summary += ["Properties"]
        for property_ in self._symbol_types.keys():
            summary += ["\t" + property_]
        summary += [""]
        summary += ["Models"]
        for model in self._models.keys():
            summary += ["\t" + model]
        summary += [""]
        return "\n".join(summary)

    def update_symbol_types(self, symbol_types):
        """
        Adds Symbol objects to the graph. If a Symbol with a given name
        is already defined on the graph, it will be replaced.

        Args:
            symbol_types (dict): dictionary of ``Symbol`` objects to add,
                keyed by symbol name
        """
        self._symbol_types.update(symbol_types)

    def remove_symbol_types(self, symbol_types):
        """
        Removes Symbol objects from the Graph. Removes
        any models that input or output this Symbol because they
        can no longer be defined without the given Symbol.

        Args:
            symbol_types (dict): dictionary of ``Symbol`` objects to remove,
                keyed by symbol name
        """
        models_to_remove = {}
        for symbol in symbol_types.keys():
            if symbol not in self._symbol_types.keys():
                raise Exception("Trying to remove a symbol that is not "
                                "currently defined.")
            if symbol_types[symbol] != self._symbol_types[symbol]:
                raise Exception("Trying to remove a symbol that is not "
                                "currently defined.")
            s1 = self._input_to_model[symbol]
            s2 = self._output_to_model[symbol]
            for m in s1:
                models_to_remove[m.name] = m
            for m in s2:
                models_to_remove[m.name] = m
            del self._symbol_types[symbol]
            del self._input_to_model[symbol]
            del self._output_to_model[symbol]
        self.remove_models(models_to_remove)

    def get_symbol_types(self):
        """
        Gets a set of all Symbol objects present on the graph.

        Returns:
            `set` of `propnet.core.symbols.Symbol`: symbols present on the graph
        """
        return set(self._symbol_types.values())

    def update_models(self, models):
        """
        Adds Model objects to the graph. If a Model with a given name
        is already defined on the graph, it will be replaced.
        The addition of a model may fail if appropriate Symbol objects
        are not already on the graph.  If any addition operation fails,
        the entire update is aborted.

        Args:
            models (dict): dictionary of ``Model`` objects to add,
                keyed by model name

        """
        added = {}
        for model in models.values():
            self._models[model.name] = model
            added[model.name] = model
            try:
                for input_set in model.evaluation_list:
                    for property_name in input_set:
                        if property_name not in self._symbol_types.keys():
                            raise KeyError(property_name)
                        self._input_to_model[property_name].add(model)
                for output_set in model.output_sets:
                    for property_name in output_set:
                        if property_name not in self._symbol_types.keys():
                            raise KeyError(property_name)
                        self._output_to_model[property_name].add(model)
            except KeyError as e:
                self.remove_models(added)
                raise KeyError("Attempted to add a model to the property "
                               "network with an unrecognized Symbol. "
                               "Add {} Symbol to the property network before "
                               "adding this model.".format(e))

    def remove_models(self, models):
        """
        Removes models from the graph.

        Args:
            models (dict): dictionary of ``Model`` objects to remove,
                keyed by model name
        """
        for model in models.keys():
            if model not in self._models.keys():
                raise Exception("Attempted to remove a model not currently "
                                "present in the graph.")
            del self._models[model]
        for s in self._input_to_model.values():
            for model in models.values():
                if model in s:
                    s.remove(model)
        for s in self._output_to_model.values():
            for model in models.values():
                if model in s:
                    s.remove(model)

    def get_models(self) -> Dict[str, Model]:
        """
        Gets a set of all Model objects present on the graph.

        Returns:
            dict: dictionary of models present on the graph, keyed by name
        """
        return {model.name: model for model in self._models.values()}

    def update_composite_models(self, composite_models):
        """
        Adds composite models (CompositeModel objects) to the graph.
        If a CompositeModel with a given name is already defined on the graph,
        it will be replaced. The addition of a model may fail if appropriate Symbol objects
        are not already on the graph. If any addition operation fails,
        the entire update is aborted.

        Args:
            composite_models (dict): dictionary of ``CompositeModel`` objects to add,
                keyed by model name
        """
        added = {}
        for model in composite_models.values():
            self._composite_models[model.name] = model
            added[model.name] = model
            for input_set in model.input_sets:
                for input_ in input_set:
                    input_ = CompositeModel.get_variable(input_)
                    if input_ not in self._symbol_types.keys():
                        self.remove_composite_models(added)
                        raise KeyError("Attempted to add a model to the property "
                                       "network with an unrecognized Symbol. "
                                       "Add {} Symbol to the property network before "
                                       "adding this model.".format(input_))

    def remove_composite_models(self, composite_models):
        """
        Removes composite models from the graph.

        Args:
            composite_models (dict): dictionary of ``CompositeModel`` objects to remove,
                keyed by model name
        """
        for model in composite_models.keys():
            if model not in self._composite_models.keys():
                raise Exception("Attempted to remove a model not currently present in the graph.")
            del self._composite_models[model]

    def get_composite_models(self):
        """
        Gets a set of all CompositeModel objects present on the graph.

        Returns:
            `set` of `propnet.core.models.CompositeModel`: composite models
            present on the graph
        """
        return set(self._composite_models.values())

    def get_networkx_graph(self, include_orphans=True):
        """
        Generates a networkX data structure representing the propnet knowledge
        graph with Symbol and Model objects as nodes and their input/output as
        directed edges.

        Args:
            include_orphans (bool): optional, ``True`` adds symbols which are not
                connected to any models to the graph object. ``False`` omits them.
                Default: ``True`` (include unconnected symbols)

        Returns:
            networkx.MultiDiGraph: NetworkX representation of knowledge graph
        """
        graph = nx.MultiDiGraph()

        # Create the abstract graph.
        for symbol in self._input_to_model:
            for model in self._input_to_model[symbol]:
                sym_type = self._symbol_types[symbol]
                graph.add_edge(sym_type, model)
        for symbol in self._output_to_model:
            for model in self._output_to_model[symbol]:
                sym_type = self._symbol_types[symbol]
                graph.add_edge(model, sym_type)

        # Add orphan nodes
        if include_orphans:
            for symbol in self._symbol_types.values():
                if symbol not in graph.nodes:
                    graph.add_node(symbol)

        # Format nodes
        # TODO: Update nx formatting with current cytoscape style
        for node in graph:
            if isinstance(node, Symbol):
                nx.set_node_attributes(graph, {node: "#43A1F8"}, "fillcolor")
                nx.set_node_attributes(graph, {node: "white"}, "fontcolor")
                nx.set_node_attributes(graph, {node: "ellipse"}, "shape")
                nx.set_node_attributes(graph, {node: node.name}, "label")
            else:
                nx.set_node_attributes(graph, {node: "orange"}, "fillcolor")
                nx.set_node_attributes(graph, {node: "white"}, "fontcolor")
                nx.set_node_attributes(graph, {node: "box"}, "shape")
                nx.set_node_attributes(graph, {node: node.name}, "label")
        return graph

    def create_file(self, filename='out.dot', draw=False, prog='dot',
                    include_orphans=False, **kwargs):
        """
        Output the propnet knowledge graph to a file using pygraphviz.

        Args:
            filename (str): optional, filename for file. Default: ``'out.dot'``
            draw (bool): optional, ``True`` renders positions for the nodes and
                edges with ``pygraphviz.AGraph.draw()``. ``False`` outputs only the
                abstract node/edge data using ``pygraphviz.AGraph.write()``.
                Default: ``False`` (write data only)
            include_orphans (bool): optional, ``True`` adds symbols which are not
                connected to any models to the graph object. ``False`` omits them.
                Default: ``True`` (include unconnected symbols)
            **kwargs: optional parameters to pygraphviz ``draw()`` or ``write()``.
        """
        nxgraph = self.get_networkx_graph(include_orphans)
        agraph = nx.nx_agraph.to_agraph(nxgraph)
        agraph.node_attr['style'] = 'filled'
        if draw:
            agraph.draw(filename, prog=prog, **kwargs)
        else:
            agraph.write(filename, **kwargs)

    # TODO: can we remove this or make it simpler?
    def calculable_properties(self, input_symbols):
        """
        Given a set of input Symbol objects, determines all new Symbol objects
        that may be calculable from the inputs. Resulting set contains
        only those new Symbol objects derivable.

        Notes:
            The result should be used with caution:
            - Models may not produce an output if their input
              conditions are not met.
            - Models may require more than one Quantity of a
              given Symbol type to generate an output.

        Args:
            input_symbols (`set` of `propnet.core.symbols.Symbol`): the set of
                Symbol objects taken as starting input properties.

        Returns:
            `set` of `propnet.core.symbols.Symbol`: the set of all Symbol objects that
            can be derived from the given input Symbols

        """
        # Set of theoretically derivable properties.
        derivable = set()

        # Set of theoretically available properties.
        working = set(input_symbols)

        # Set of all models that could produce output.
        all_models = set()
        c_models = set()
        for sym in input_symbols:
            for model in self._input_to_model[sym]:
                all_models.add(model)
                c_models.add(model)

        to_add = set()
        to_remove = set()

        has_changed = True

        # TODO: revisit this and cleanup, looks too complicated
        while has_changed:
            # Add any new models to investigate.
            for m in to_add:
                c_models.add(m)
            to_add = set()
            # Remove any models that can't augment the Symbols set.
            for m in to_remove:
                c_models.remove(m)
            to_remove = set()
            # Check if any models generate new Symbol objects as outputs.
            has_changed = False
            for model in c_models:
                # Check if model can add a new Symbols
                can_contribute = False
                for output in model.all_outputs:
                    if output not in working:
                        can_contribute = True
                        break
                if not can_contribute:
                    to_remove.add(model)
                    continue
                # Check if model has all constraint Symbols provided.
                has_inputs = True
                for s in model.constraint_symbols:
                    if s not in working:
                        has_inputs = False
                        break
                if not has_inputs:
                    continue
                # Check if any model input sets are met.
                paired_sets = zip(model.input_sets, model.output_sets)
                for input_set, output_set in paired_sets:
                    has_inputs = True
                    for s in input_set:
                        if s not in working:
                            has_inputs = False
                            break
                    if not has_inputs:
                        continue
                    # Check passed -- add model outputs to the available properties.
                    #              -- add any new models working with these newly available properties
                    for s in output_set:
                        if s not in working:
                            for new_model in self._input_to_model[s]:
                                if new_model not in all_models:
                                    all_models.add(new_model)
                                    to_add.add(new_model)
                            working.add(s)
                            derivable.add(self._symbol_types[s])
                            has_changed = True

        return derivable

    def required_inputs_for_property(self, target):
        """
        Determines all potential paths leading to a given symbol
        object. Answers the question: What sets of properties are
        required to calculate this given property?

        Paths are represented as a series of models and required
        input Symbol objects. Paths can be searched to determine
        specifically how to get from one property to another.

        Notes:
            Warning: Method indicates sets of Symbol objects required
            to calculate the property. It does not indicate how
            many of each Symbol is required. It does not guarantee
            that supplying Quantity objects of these types will result
            in a new Symbol output as conditions/constraints may
            not be met.

        Args:
            target (Symbol): desired target symbol
        Returns:
            SymbolTree: pathways to target property represented as a tree
        """
        head = TreeElement(None, {target}, None, None)
        self._tree_builder(head)
        return SymbolTree(head)

    def _tree_builder(self, tree_to_expand: TreeElement):
        """
        Recursive helper method to build a SymbolTree. Fills in
        the children of to_expand by all possible model
        substitutions.

        Args:
            tree_to_expand (TreeElement): element that will be expanded in place
        """
        # Get set of symbols that no longer need to be replaced and
        # symbols that are candidates for replacement.
        replaced_symbols = set()    # set of all symbols already replaced.
                                    # equal to all parents' minus expand's symbols.
        parent = tree_to_expand.parent
        while parent is not None:
            replaced_symbols.update(parent.inputs)
            parent = parent.parent
        replaced_symbols -= tree_to_expand.inputs
        candidate_symbols = tree_to_expand.inputs - replaced_symbols

        # Attempt to replace candidate_symbols
        # Replace them with inputs to models that output the candidate_symbols.
        # Store replacements.
        outputs = []
        prev = defaultdict(list)
        for symbol in candidate_symbols:
            c_models = self._output_to_model[symbol]
            for model in c_models:
                parent = tree_to_expand.parent
                parent: TreeElement
                can_continue = True
                while parent is not None:
                    if parent.m is model:
                        can_continue = False
                        break
                    parent = parent.parent
                if not can_continue:
                    continue
                for input_set, output_set in zip(model.input_sets, model.output_sets):
                    can_continue = True
                    for input_symbol in input_set:
                        if input_symbol in replaced_symbols:
                            can_continue = False
                            break
                    if not can_continue:
                        continue
                    input_set = input_set | model.constraint_symbols
                    new_types = (tree_to_expand.inputs - output_set)
                    new_types.update(input_set)
                    new_types = {self._symbol_types[x] for x in new_types}
                    if new_types in prev[model]:
                        continue
                    prev[model].append(new_types)
                    new_element = TreeElement(model, new_types, tree_to_expand, None)
                    self._tree_builder(new_element)
                    outputs.append(new_element)

        # Add outputs to children and fill in their elements.
        tree_to_expand.children = outputs

    # TODO: can we remove this?
    def get_paths(self, start_property, end_property):
        """
        Returns all paths between two properties.

        Notes:
            This method is very computationally expensive in its current implementation.
            We are actively seeking more efficient ways of calculating pathways between
            properties.
        Args:
            start_property (Symbol): starting Symbol type
            end_property (Symbol): ending Symbol type
        Returns:
            `list` of `propnet.core.provenance.SymbolPath`: list enumerating the features
            of all paths
        """
        tree = self.required_inputs_for_property(end_property)
        return tree.get_paths_from(start_property)

    def get_degree_of_separation(self, start_property: Union[str, Symbol],
                                 end_property: Union[str, Symbol]) -> Union[int, None]:
        """
        Determines the minimum number of models separating two properties (symbols)
        on the propnet knowledge graph.

        Notes:
            Because the propnet knowledge graph is directed, A->B may have a
            valid pathway, but B->A may not.

        Args:
            start_property (`str` or `Symbol`): starting/input property
            end_property (`str` or `Symbol`): ending/derived property

        Returns:
            `int` or `None`: the minimum number of models separating the two properties,
            where ``0`` indicates the starting and ending properties are equal and ``None``
            indicates the two properties are not connected by any models.
        """

        # TODO: Would it be faster to use networkx?
        # Ensure we have the properties in the graph.
        if start_property not in self._symbol_types.keys():
            raise ValueError("Symbol not found: " + str(start_property))
        if end_property not in self._symbol_types.keys():
            raise ValueError("Symbol not found: " + str(end_property))
        # Coerce types into actual Symbol objects.
        start_property = self._symbol_types[start_property]
        end_property = self._symbol_types[end_property]
        # Take care of case where start and end properties are the same
        if start_property == end_property:
            return 0
        # Setup helper datastructures
        visited = set()             # all properties visited
        visited.add(start_property)
        to_visit = deque()          # all properties to visit in the current depth
        to_visit.append(start_property)
        to_visit_next = deque()     # all properties to visit in the next depth
        depth_count = 0
        found = False
        # Search for the target property
        while True:
            while len(to_visit) != 0 and not found:
                visiting = to_visit.popleft()
                extending_models = self._input_to_model[visiting]
                for model in extending_models:
                    if found: break
                    for output_set in model.output_sets:
                        if found: break
                        for property_name in output_set:
                            connection = self._symbol_types[property_name]
                            if connection == end_property:
                                found = True
                                break
                            if connection in visited:
                                continue
                            visited.add(connection)
                            to_visit_next.append(connection)
            depth_count += 1
            if found or len(to_visit_next) == 0:
                break
            while len(to_visit_next) != 0:
                to_visit.append(to_visit_next.popleft())
        if found:
            return depth_count
        if not found:
            return None

    @staticmethod
    def generate_input_sets(props, this_quantity_pool):
        """
        Generates all unique combinations of quantities given a list of needed
        symbols/properties names and a pool of quantities to choose from.

        Args:
            props (`list` of `str` or `propnet.core.symbols.Symbol`): desired properties
                in input set
            this_quantity_pool (dict): quantity pool, as a dictionary of sets of quantities
                keyed by their Symbol or symbol name
        Yields:
            `tuple` of `Quantity`: tuple of length ``len(props)`` containing Quantity objects
            corresponding to each symbol in ``props``.
        """
        aggregated_symbols = []
        for prop in props:
            if prop not in this_quantity_pool.keys():
                return
            aggregated_symbols.append(this_quantity_pool[prop])
        yield from product(*aggregated_symbols)

    @staticmethod
    def get_input_sets_for_model(model, new_quantities, old_quantities):
        """
        Generates all valid input sets for a given model, containing at least
        one Quantity from ``new_quantities`` with the remainder drawn from
        ``old_quantities``.

        Args:
            model (Model): model for which to evaluate valid input sets
            new_quantities (dict): quantities generated
                during the most recent iteration of the evaluation loop,
                as lists of Quantity objects keyed by symbol
            old_quantities (dict): quantities generated
                in previous iterations of the evaluation loop,
                as lists of Quantity objects keyed by symbol
        Returns:
            Tuple[iterator, int]: returns tuple containing:

            - iterator yielding input sets as tuples of Symbol objects
            - integer corresponding to the number of items in the iterator
        """

        all_input_sets = []
        OLD = 0
        NEW = 1
        source_map = {OLD: old_quantities, NEW: new_quantities}
        n_input_sets = 0

        for symbols_to_evaluate in model.evaluation_list:
            sources_by_symbol = []
            for symbol in symbols_to_evaluate:
                symbol_sources = []
                if symbol in old_quantities.keys():
                    symbol_sources.append(OLD)
                if symbol in new_quantities.keys():
                    symbol_sources.append(NEW)
                sources_by_symbol.append(symbol_sources)
            source_combinations = list(product(*sources_by_symbol))

            for sources in source_combinations:
                if all(s == OLD for s in sources):
                    continue
                symbols_to_combine = [source_map[source][symbol]
                                      for source, symbol in zip(sources, symbols_to_evaluate)]
                n_input_sets += np.prod([len(v) for v in symbols_to_combine])
                all_input_sets.append(product(*symbols_to_combine))

        return chain.from_iterable(all_input_sets), n_input_sets

    def generate_models_and_input_sets(self, new_quantities, quantity_pool):
        """
        Produces all input sets for all models on the graph that contain at least
        one Quantity from ``new_quantities``.

        Args:
            new_quantities (`list` of `BaseQuantity`): list of new quantities from which
                to derive new input sets
            quantity_pool (dict): dict of Quantity sets,
                keyed by symbol, from which to draw additional quantities
                for model inputs
        Returns:
            Tuple[iterator, int]: tuple that contains:

            - an iterator containing models and input sets as tuples containing:

                - ``Model`` instance for which the input set is valid
                - tuple of Quantity objects representing the input set

            - an integer representing the total number of input sets the iterator
        """
        models_and_input_sets = []
        n_total_input_sets = 0
        new_qs_by_symbol = defaultdict(list)
        for quantity in new_quantities:
            new_qs_by_symbol[quantity.symbol].append(quantity)

        candidate_models = set()
        for symbol in new_qs_by_symbol.keys():
            for model in self._input_to_model[symbol]:
                candidate_models.add(model)

        for model in candidate_models:
            input_sets, n_input_sets = self.get_input_sets_for_model(
                model, new_qs_by_symbol, quantity_pool)
            if n_input_sets > 0:
                models_and_input_sets.append(zip(repeat(model, n_input_sets), input_sets))
                n_total_input_sets += n_input_sets

        return chain.from_iterable(models_and_input_sets), n_total_input_sets

    def derive_quantities(self, new_quantities, quantity_pool=None,
                          allow_model_failure=True, timeout=None):
        """
        Derives new quantities using the models on the knowledge graph using at least one quantity
        from ``new_quantities`` and the remainder from either ``new_quantities`` or the specified
        ``quantity_pool`` as inputs to the models.

        Args:
            new_quantities (`list` of `BaseQuantity`): list of quantities which to
                consider as new inputs to models
            quantity_pool (`dict` or `None`): optional, dict of lists of ``BaseQuantity`` objects,
                keyed by their ``Symbol`` from which to draw additional quantities
                for model inputs. Default: ``None`` (no pool)
            allow_model_failure (bool): ``True`` allows graph evaluation to
                continue if an Exception is thrown during model evaluation for
                violation of constraints or any other reason. ``False`` will
                throw any Exception encountered during model evaluation.
                Default: ``True`` (ignore exceptions)
            timeout (`int` or `None`): number of seconds to allow for a model to evaluate.
                After that time, model evaluation will be canceled and deemed
                failed. ``None`` allows for infinite evaluation time. Default: ``None`` (no limit)

        Returns:
            Tuple[list, dict]: returns a list and dict in a tuple:

                - derived quantities as `list` of `BaseQuantity`
                - quantity pool augmented with quantities from ``new_quantities``
        """
        # Update quantity pool
        quantity_pool = quantity_pool or defaultdict(list)

        # Generate all of the models and input sets to be evaluated

        logger.info("Generating models and input sets for %s", new_quantities)

        models_and_input_sets, n_input_sets = self.generate_models_and_input_sets(
            new_quantities, quantity_pool)

        # TODO: Maybe we should do this in evaluate() instead of here
        for quantity in new_quantities:
            quantity_pool[quantity.symbol].append(quantity)

        # input_tuples = [(v[0], v[1:]) for v in models_and_input_sets]

        # The code below doesn't eliminate many and will be caught by cyclic filter
        # after evaluation. This is usually only important if the model we'd be
        # re-evaluating takes a long time, which the majority of our models do not
        # take a long time...can we move this into the generation step or just before
        # evaluation so we don't have to break the generator open?

        # inputs_to_calculate = list(filter(Graph._generates_noncyclic_output,
        #                                   input_tuples))

        # Evaluate model for each input set and add new valid quantities
        if not self._parallel:
            with Timer('_graph_evaluation'):
                added_quantities, model_timings = Graph._run_serial(models_and_input_sets,
                                                                    allow_model_failure=allow_model_failure,
                                                                    timeout=timeout)
        else:
            if self._executor is None:
                self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=self._max_workers)
            with Timer('_graph_evaluation'):
                added_quantities, model_timings = Graph._run_parallel(self._executor, self._max_workers,
                                                                      models_and_input_sets,
                                                                      n_input_sets,
                                                                      allow_model_failure=allow_model_failure,
                                                                      timeout=timeout)

        self._append_timing_result(model_timings)
        self._graph_timings = {k: v for k, v in timings['_graph_evaluation'].items() if k != 'timings'}
        self._model_timings = copy.deepcopy(timings['_graph_evaluation']['timings'])

        return added_quantities, quantity_pool

    @staticmethod
    def _generates_noncyclic_output(input_set):
        """
        Determines if an input set of model and input quantities will generate at least one output that
        is non-cyclic, meaning the output does not have itself as an input in its provenance.

        Args:
            input_set (tuple): input set to evaluate for cyclic outputs as a tuple of Model
                and list of BaseQuantity objects.

        Returns:
            bool: ``True`` if at least one output of the model is non-cyclic. ``False`` if all outputs
            are cyclic.
        """
        model, inputs = input_set
        input_symbols = set(v.symbol for v in inputs)
        outputs = set()
        for s in model.connections:
            if set(model.map_variables_to_symbols(s['inputs'])) == input_symbols:
                outputs = outputs.union(model.map_variables_to_symbols(s['outputs']))

        model_in_all_trees = all(input_q.provenance.model_is_in_tree(model) or
                                 model == input_q.provenance.model
                                 for input_q in inputs)

        symbol_in_all_trees = all(all(input_q.provenance.symbol_is_in_tree(output)
                                      for input_q in inputs)
                                  for output in outputs)

        return not (model_in_all_trees and symbol_in_all_trees)

    @staticmethod
    def _run_serial(models_and_input_sets, allow_model_failure=True, timeout=None):
        """
        Evaluates a list of input sets serially.

        Args:
            models_and_input_sets (`list` of `tuple`): input sets to evaluate as a list of tuples containing
                a Model and a list of BaseQuantity objects.
            allow_model_failure (bool): optional, ``True`` suppresses exceptions raised during model evaluation.
                ``False`` throws them as they are raised. Default: ``True`` (no exceptions raised)
            timeout (int): optional, number of seconds after which to timeout model evaluation.
                Default: ``None`` (no limit)

        Returns:
            `list` of `BaseQuantity`: output quantities from input set evaluation.
        """
        outputs = []
        model_timings = []
        for model_and_input_set in models_and_input_sets:
            result = Graph._evaluate_model(model_and_input_set,
                                           allow_failure=allow_model_failure,
                                           timeout=timeout)

            if isinstance(result, tuple) and isinstance(result[0], Exception):
                model, inputs = result[1]
                exception = result[0]
                raise Exception("Encountered error with model {}"
                                " and input set {}:\n"
                                "Exception raised: {}: {}".format(model.name, inputs,
                                                                  type(exception).__name__, exception))
            else:
                q_list, timing_ = result
                outputs.extend(q_list)
                model_timings.append(timing_)
        return outputs, model_timings

    @staticmethod
    def _run_parallel(executor, n_workers, models_and_input_sets,
                      n_total_input_sets,
                      allow_model_failure=True,
                      timeout=None):
        """
        Evaluate a list of input sets in parallel.

        Args:
            executor (concurrent.futures.ProcessPoolExecutor): executor for input sets
            n_workers (int): number of processes used by executor
            models_and_input_sets (`list` of `tuple`): input sets to evaluate as a list of tuples containing
                a Model and a list of BaseQuantity objects.
            allow_model_failure (bool): optional, ``True`` suppresses exceptions raised during model evaluation.
                ``False`` throws them as they are raised. Default: ``True`` (no exceptions raised)
            timeout (int): optional, number of seconds after which to timeout model evaluation.
                Default: ``None`` (no limit)

        Returns:
            `list` of `BaseQuantity`: output quantities from input set evaluation.
        """
        func = partial(Graph._evaluate_model,
                       allow_failure=allow_model_failure,
                       timeout=timeout)
        chunk_size = min(int(n_total_input_sets / n_workers) + 1, 200)
        results = executor.map(func, models_and_input_sets, chunksize=chunk_size)
        outputs = []
        model_timings = []
        for result in results:
            if isinstance(result, tuple) and isinstance(result[0], Exception):
                model, inputs = result[1]
                exception = result[0]
                raise Exception("Encountered error with model {}"
                                " and input set {}:\n"
                                "Exception raised: {}: {}".format(model.name, inputs,
                                                                  type(exception).__name__, exception))
            else:
                q_list, timing_ = result
                outputs.extend(q_list)
                model_timings.append(timing_)
        return outputs, model_timings

    @staticmethod
    def _evaluate_model(model_and_input_set, allow_failure=True, timeout=None):
        """
        Evaluates an input set.

        Notes:
            The exception is returned instead of thrown because in parallel, the exception will be suppressed
            by the map() function used to execute the model evaluation in parallel. Additionally, the timings
            are returned because they do not sum when run in parallel because timing is executed on different
            processors.

        Args:
            model_and_input_set (tuple): input set to evaluate as a tuple containing
                a Model and a list of BaseQuantity objects.
            allow_failure (bool): optional, ``True`` suppresses exceptions raised during model evaluation.
                ``False`` throws them as they are raised. Default: ``True`` (no exceptions raised)

        Returns:
            Tuple[list, dict]: tuple containing the following data:

                - list of quantities calculated from model. If model failed and allow_failure = True, will
                  return an empty list. If allow_failure = False, will return a list with a tuple
                  (Exception, input_set) as its only element.
                - dictionary of timing data for this model
        """
        # from chronic import Timer as Timer_
        # from chronic import timings as timings_
        model, inputs = model_and_input_set
        input_dict = {q.symbol: q for q in inputs}
        logger.info('Evaluating %s with input %s', model, input_dict)

        try:
            with Timer(model.name):
                with Timeout(seconds=timeout,
                             error_message=f"Evaluation took longer than specified timeout"):
                    result = model.evaluate(input_dict,
                                            allow_failure=allow_failure)
        except TimeoutError:
            if allow_failure:
                result = {'successful': False,
                          'message': 'Model evaluation timed out for {}'.format(model.name)}
            else:
                return TimeoutError('Model evaluation timed out for {}'.format(model.name)), model_and_input_set
        except Exception as ex:
            # If we get an exception, then allow_failure should be False
            # or we got some other exception we need to know about
            return ex, model_and_input_set
        # TODO: Maybe provenance should be done in evaluate?

        success = result.pop('successful')
        if success:
            out = [v for v in result.values() if not v.is_cyclic()]
        else:
            logger.info("Model evaluation unsuccessful %s",
                        result['message'])
            out = []
        timing_data = {model.name: {k: v for k, v in timings[model.name].items()}}
        timings.pop(model.name)
        return out, timing_data

    def evaluate(self, material, allow_model_failure=True, timeout=None):
        """
        Given a Material object as input, creates a new Material object
        to include all derivable properties for that material.

        Notes:
            Model timeout does not work on non-Unix machines based on the implementation. ``timeout`` will
            be ignored on these machines.

        Args:
            material (Material): a material whose properties will be expanded
            allow_model_failure (bool): optional, ``True`` continues with graph evaluation
                if a model fails. ``False`` throws the exception.
                Default: ``True`` (ignore failed models)
            timeout (`int` or `None`): optional, number of seconds after which model evaluation should
                quit. This is to cut off long-running models.
                Default: ``None`` (infinite evaluation time)
        Returns:
            Material: material object containing all properties, derived + original inputs
        """
        logger.debug("Beginning evaluation")

        # Generate initial quantity set and pool
        new_quantities = material.get_quantities()
        quantity_pool = None

        # Derive new Quantities
        # Loop util no new Quantity objects are derived.
        logger.debug("Beginning main loop with quantities %s", new_quantities)
        while new_quantities:
            new_quantities, quantity_pool = self.derive_quantities(
                new_quantities, quantity_pool,
                allow_model_failure=allow_model_failure,
                timeout=timeout)

        new_material = Material()
        new_material._quantities_by_symbol = quantity_pool
        return new_material

    def evaluate_composite(self, material, allow_model_failure=True,
                           allow_composite_model_failure=True,
                           timeout=None):
        """
        Given a CompositeMaterial object as input, creates a new CompositeMaterial
        object to include all derivable properties for that material.

        Args:
            material (CompositeMaterial): material whose properties
                will be expanded
            allow_model_failure (bool): optional, ``True`` continues with graph evaluation
                if a non-CompositeModel fails. ``False`` throws the exception.
                Default: ``True`` (ignore failed non-composite models)
            allow_composite_model_failure (bool): ``True`` continues with graph evaluation
                if a CompositeModel fails. ``False`` throws the exception.
                Default: ``True`` (ignore failed composite models)
            timeout (`int` or `None`): optional, number of seconds after which non-CompositeModel
                evaluation should quit. This is to cut off long-running models.
                Default: ``None`` (infinite evaluation time)

        Returns:
            CompositeMaterial: composite material object containing all properties, derived + original inputs
        """

        # TODO: Let's parallelize this eventually. It's not immediately obvious to me
        #       the best loop to parallelize, so will wait until we have more composite
        #       models to evaluate.

        if not isinstance(material, CompositeMaterial):
            raise Exception("material provided is not a CompositeMaterial: " + str(type(material)))

        # Evaluate material's sub-materials
        evaluated_materials = list()
        for m in material.materials:
            logger.debug("Evaluating sub-material: " + str(id(m)))
            if isinstance(m, CompositeMaterial):
                evaluated_materials.append(self.evaluate_composite(m, allow_model_failure=allow_model_failure,
                                                                   timeout=timeout))
            else:
                evaluated_materials.append(self.evaluate(m, allow_model_failure=allow_model_failure,
                                                         timeout=timeout))

        # Run all CompositeModels in the graph on this SuperMaterial if
        # a material mapping can be established.  Store any derived quantities.
        all_quantities = defaultdict(set)
        for (k, v) in material._quantities_by_symbol:
            all_quantities[k].add(v)

        to_return = CompositeMaterial(evaluated_materials)
        to_return._quantities_by_symbol = all_quantities

        logger.debug("Evaluating CompositeMaterial")

        for model in self._composite_models.values():

            logger.debug("\tEvaluating Model: " + model.name)

            # Establish material mappings for the given input set.

            mat_mappings = model.gen_material_mappings(to_return.materials)

            # Avoid ambiguous or impossible mappings, at least for now.
            if len(mat_mappings) != 1:
                continue

            mat_mapping = mat_mappings[0]

            # Go through input sets

            for property_input_sets in model.evaluation_list:

                logger.debug("\t\tGenerating input sets for: " + str(property_input_sets))

                # Create a quantity pool from the appropriate materials.
                # Modify inputs for use in generate_input_sets

                temp_pool = defaultdict(set)
                combined_list = []
                mat_list = []
                symbol_list = []
                for item in property_input_sets:
                    combined_list.append(item)
                    mat_list.append(CompositeModel.get_material(item))
                    symbol_list.append(CompositeModel.get_variable(item))
                for i in range(0,len(mat_list)):
                    if mat_list[i] is None:     # Draw symbol from the CompositeMaterial
                        mat = to_return
                    else:
                        mat = mat_mapping[mat_list[i]]
                    for q in mat._quantities_by_symbol[symbol_list[i]]:
                        temp_pool[combined_list[i]].add(q)
                input_sets = self.generate_input_sets(combined_list, temp_pool)

                for input_set in input_sets:

                    logger.debug("\t\t\tEvaluating input set: " + str(input_set))

                    # Check if input_set can be evaluated -- input_set must pass the necessary model constraints
                    if not model.check_constraints(input_set):
                        logger.debug("\t\t\tInput set failed -- did not pass model constraints.")
                        continue

                    # Try to evaluate input_set:
                    evaluate_set = dict(zip(combined_list, input_set))
                    output = model.evaluate(
                        evaluate_set, allow_failure=allow_composite_model_failure)
                    success = output.pop('successful')
                    if not success:
                        logger.debug("\t\t\tInput set failed -- did not produce a successful output.")
                        continue

                    # input_set led to output from the Model -- add output to the CompositeMaterial

                    logger.debug("\t\t\tInput set produced successful output.")
                    for symbol, quantity in output.items():
                        st = self._symbol_types.get(symbol)
                        if not st:
                            raise ValueError(
                                "Symbol type {} not found".format(symbol))
                        q = QuantityFactory.create_quantity(st, quantity)
                        to_return._quantities_by_symbol[st].add(q)
                        logger.debug("\t\t\tNew output: " + str(q))

        # Evaluate the CompositeMaterial's quantities and return the result.
        mappings = self.evaluate(to_return)._quantities_by_symbol
        to_return._quantities_by_symbol = mappings
        return to_return

    def clear_statistics(self):
        """
        Clears model evaluation timings.

        Notes:
            If you are using the ``chronic.Timer`` module for timing outside this Graph object,
            this function will clear your timers causing an error if the Timer objects
            are currently running.

        """
        self._graph_timings = None
        self._model_timings = None
        clear_timings()

    @property
    def model_evaluation_statistics(self):
        """
        Compiles a pandas DataFrame containing statistics on how
        many times each model was evaluated, average time per model,
        and the total time taken for that model.

        Returns:
            pandas.DataFrame: model calculation statistics
        """

        rows = [{'Model Name': model,
                 'Total Evaluation Time /s': stats['total_elapsed'],
                 'Average Evaluation Time /s': stats['average_elapsed'],
                 'Number of Evaluations': stats['count']}
                for model, stats in self._model_timings.items()]

        return DataFrame(rows, columns=['Model Name',
                                        'Total Evaluation Time /s',
                                        'Number of Evaluations',
                                        'Average Evaluation Time /s'])

    @staticmethod
    def _append_timing_result(model_timings):
        """
        Adds model timings collected from parallel processes to
        the timings module in this thread/process.

        Args:
            model_timings (`list` of `dict`): list of model timings returned from evaluation
        """
        if 'timings' not in timings['_graph_evaluation']:
            timings['_graph_evaluation']['timings'] = dict()

        ge_timings = timings['_graph_evaluation']['timings']
        for timing_ in model_timings:
            for model_name, stats in timing_.items():
                if model_name not in ge_timings.keys():
                    ge_timings[model_name] = {'total_elapsed': 0.,
                                              'count': 0,
                                              'average_elapsed': 0.}
                ge_timings[model_name]['total_elapsed'] += stats['total_elapsed']
                ge_timings[model_name]['count'] += stats['count']

        for model_name, stats in ge_timings.items():
            stats['average_elapsed'] = stats['total_elapsed'] / stats['count']




