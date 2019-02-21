"""
Module containing classes and methods for graph functionality in Propnet code.
"""

import logging
from collections import defaultdict
from itertools import product
from chronic import Timer, timings
from pandas import DataFrame
from collections import deque
import concurrent.futures
from functools import partial
from multiprocessing import cpu_count

import networkx as nx

from propnet.core.materials import CompositeMaterial
from propnet.core.materials import Material
from propnet.core.models import Model, CompositeModel
from propnet.core.quantity import QuantityFactory
from propnet.core.provenance import SymbolTree, TreeElement
from propnet.models import COMPOSITE_MODEL_DICT
from propnet.models import DEFAULT_MODEL_DICT
from propnet.symbols import Symbol, DEFAULT_SYMBOLS

from typing import Set, Dict, Union

logger = logging.getLogger(__name__)


class Graph(object):
    """
    Class containing methods for creating and interacting with a
    Property Network.
    The Property Network contains a set of Node namedtuples with
    connections stored as directed edges between the nodes.
    Upon initialization a base graph is constructed consisting of all
    valid SymbolTypes and Models found in surrounding folders. These are
    Symbol and Model node_types respectively. Connections are formed
    between the nodes based on given inputs and outputs of the models.
    At this stage the graph represents a symbolic web of properties
    without any actual input values.
    Materials and Properties / Conditions can be added at runtime using
    appropriate support methods. These methods dynamically create
    additional PropnetNodes and edges on the graph of Material and
    Quantity node_types respectively.
    Given a set of Materials and Properties / Conditions, the symbolic
    web of properties can be utilized to predict values of connected
    properties on demand.
    Attributes:
        _symbol_types ({str: Symbol}): data structure mapping Symbol
            name to Symbol object.
        _models ({str: Model}): data structure mapping Model name to
            Model object.
        _input_to_model ({Symbol: {Model}}): data structure mapping
            Symbol inputs to a set of corresponding Model objects that
            take that Symbol as an input.
        _output_to_model ({Symbol: {Model}}): data structure mapping
            Symbol outputs to a set of corresponding Model objects that
            produce that Symbol as an output.
    *** Dictionaries can be searched by supplying Symbol objects or
        Strings as to their names.
    """

    def __init__(self,
                 models: Dict[str, Model]=None,
                 composite_models: Dict[str, CompositeModel]=None,
                 symbol_types: Dict[str, Symbol]=None) -> None:
        """
        Creates a Graph instance
        """

        # set our defaults if no models/symbol types supplied
        symbol_types = symbol_types or DEFAULT_SYMBOLS

        # create the graph
        self._symbol_types = dict()
        self._models = dict()
        self._composite_models = dict()
        self._input_to_model = defaultdict(set)
        self._output_to_model = defaultdict(set)
        self._timings = None                        # type:Dict[,]
        self._n_workers = None

        if symbol_types:
            self.update_symbol_types(symbol_types)

        if models is None:
            self.update_models(DEFAULT_MODEL_DICT)
        else:
            self.update_models(models)

        if composite_models is None:
            self.update_composite_models(COMPOSITE_MODEL_DICT)
        else:
            self.update_composite_models(composite_models)

    def __str__(self):
        """
        Returns a full summary of the graph in terms of the SymbolTypes,
        Symbols, Materials, and Models that it contains. Connections are
        shown as nesting within the printout.
        Returns:
            (str) representation of this Graph object.
        """
        summary = ["Propnet Printout", ""]
        summary += ["Properties"]
        for property in self._symbol_types.keys():
            summary += ["\t" + property]
        summary += [""]
        summary += ["Models"]
        for model in self._models.keys():
            summary += ["\t" + model]
        summary += [""]
        return "\n".join(summary)

    def update_symbol_types(self, symbol_types):
        """
        Add / redefine user-defined symbol types to the graph. If the
        input, symbol_types, includes keys in self._symbol_types,
        they are redefined.
        Args:
            symbol_types ({name: Symbol}): symbol types to add
        Returns:
            None
        """
        for (k, v) in symbol_types.items():
            self._symbol_types[k] = v

    def remove_symbol_types(self, symbol_types):
        """
        Removes user-defined Symbol objects to the Graph. Removes
        any models that input or output this Symbol because they
        are no longer defined without the given symbol_types.
        Args:
            symbol_types ({name:Symbol}): symbol types to remove
        Returns:
            None
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
        Getter method, returns a set of all Symbol objects
        present on the graph.
        Returns ({Symbol}):
            set of symbols present on the graph
        """
        to_return = set()
        for s in self._symbol_types.values():
            to_return.add(s)
        return to_return

    def update_models(self, models):
        """
        Add / redefine user-defined models to the graph. If the input,
        models, includes keys in self._models, they are redefined.
        The addition of a model may fail if appropriate Symbol objects
        are not already on the graph.  If any addition operation fails,
        the entire update is aborted.
        Args:
            models ({name: Model}): Instances of the model class
        Returns:
            None
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
        Remove user-defined models from the Graph.
        Args:
            models ({name: Model}): Instances of the model class
        Returns:
            None
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
        Getter method, returns a set of all model objects present
        on the graph.
        Returns ({Model}):
            set of models in the graph
        """
        to_return = dict()
        for model in self._models.values():
            to_return[model.name] = model
        return to_return

    def update_composite_models(self, super_models):
        """
        Add / redefine user-defined super_models to the graph.
        If the input, super_models, includes keys in self._super_models, they are redefined.
        The addition of a super_model may fail if appropriate Symbol objects are not already on the graph.
        If any addition operation fails, the entire update is aborted.
        Args:
            super_models (dict<str, SuperModel>): Instances of the SuperModel class
        Returns:
            None
        """
        added = {}
        for model in super_models.values():
            self._composite_models[model.name] = model
            added[model.name] = model
            for input_set in model.input_sets:
                for input in input_set:
                    input = CompositeModel.get_symbol(input)
                    if input not in self._symbol_types.keys():
                        raise KeyError("Attempted to add a model to the property "
                               "network with an unrecognized Symbol. "
                               "Add {} Symbol to the property network before "
                               "adding this model.".format(input))

    def remove_composite_models(self, super_models):
        """
        Remove user-defined models from the Graph.
        Args:
            super_models (dict<str, SuperModel>): Instances of the SuperModel class
        Returns:
            None
        """
        for model in super_models.keys():
            if model not in self._composite_models.keys():
                raise Exception("Attempted to remove a model not currently present in the graph.")
            del self._composite_models[model]

    def get_composite_models(self):
        """
        Getter method, returns a set of all model objects present on the graph.
        Returns:
            (set<Model>)
        """
        to_return = set()
        for model in self._composite_models.values():
            to_return.add(model)
        return to_return

    def get_networkx_graph(self, include_orphans=True):
        """
        Generates a networkX data structure representing the property
        network and returns this object.
        Returns:
            (networkX.multidigraph)
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
        Output the graph to a file
        Args:
            filename (str): filename for file
            draw (bool): whether to draw or write file
            include_orphans (bool): whether to include orphan symbols
                in graph output
            **kwargs (kwargs): kwargs to draw or write
        Returns:
            None
        """
        nxgraph = self.get_networkx_graph(include_orphans)
        agraph = nx.nx_agraph.to_agraph(nxgraph)
        agraph.node_attr['style'] = 'filled'
        if draw:
            agraph.draw(filename, prog=prog, **kwargs)
        else:
            agraph.write(filename, **kwargs)


    # TODO: can we remove this?
    def calculable_properties(self, property_type_set):
        """
        Given a set of Symbol objects, returns all new Symbol objects
        that may be calculable from the inputs. Resulting set contains
        only those new Symbol objects derivable.
        The result should be used with caution:
            1) Models may not produce an output if their input
                conditions are not met.
            2) Models may require more than one Quantity of a
                given Symbol type to generate an output.
        Args:
            property_type_set ({Symbol}): the set of Symbol objects
                taken as starting properties.
        Returns:
            (({Symbol}, {Model})) the set of all Symbol objects that
                can be derived from the property_type_set, the set of
                all Model objects that are used in deriving the new
                Symbol objects.
        """
        # Set of theoretically derivable properties.
        derivable = set()

        # Set of theoretically available properties.
        working = set()
        for property_type in property_type_set:
            working.add(property_type)

        # Set of all models that could produce output.
        all_models = set()
        c_models = set()
        for property_type in property_type_set:
            for model in self._input_to_model[property_type]:
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
                for s in model.constraint_properties:
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

    def required_inputs_for_property(self, property):
        """
        Determines all potential paths leading to a given symbol
        object. Answers the question: What sets of properties are
        required to calculate this given property?
        Paths are represented as a series of models and required
        input Symbol objects. Paths can be searched to determine
        specifically how to get from one property to another.
        Warning: Method indicates sets of Symbol objects required
            to calculate the property.  It does not indicate how
            many of each Symbol is required. It does not guarantee
            that supplying Quantities of these types will result
            in a new Symbol output as conditions / assumptions may
            not be met.
        Returns:
            propnet.core.utils.SymbolTree
        """
        head = TreeElement(None, {property}, None, None)
        self._tree_builder(head)
        return SymbolTree(head)

    def _tree_builder(self, to_expand: TreeElement):
        """
        Recursive helper method to build a SymbolTree.  Fills in
        the children of to_expand by all possible model
        substitutions.
        Args:
            to_expand: (TreeElement) element that will be expanded
        Returns:
            None
        """
        # Get set of symbols that no longer need to be replaced and
        # symbols that are candidates for replacement.
        replaced_symbols = set()    # set of all symbols already replaced.
                                    # equal to all parents' minus expand's symbols.
        parent = to_expand.parent
        while parent is not None:
            replaced_symbols.update(parent.inputs)
            parent = parent.parent
        replaced_symbols -= to_expand.inputs
        candidate_symbols = to_expand.inputs - replaced_symbols

        # Attempt to replace candidate_symbols
        # Replace them with inputs to models that output the candidate_symbols.
        # Store replacements.
        outputs = []
        prev = defaultdict(list)
        for symbol in candidate_symbols:
            c_models = self._output_to_model[symbol]
            for model in c_models:
                parent = to_expand.parent
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
                    input_set = input_set | model.constraint_properties
                    new_types = (to_expand.inputs - output_set)
                    new_types.update(input_set)
                    new_types = {self._symbol_types[x] for x in new_types}
                    if new_types in prev[model]:
                        continue
                    prev[model].append(new_types)
                    new_element = TreeElement(model, new_types, to_expand, None)
                    self._tree_builder(new_element)
                    outputs.append(new_element)

        # Add outputs to children and fill in their elements.
        to_expand.children = outputs

    # TODO: can we remove this?
    def get_paths(self, start_property, end_property):
        """
        Returns all Paths
        Args:
            start_property: (Symbol) starting Symbol type
            end_property: (Symbol) ending Symbol type
        Returns:
            (list<SymbolPath>) list enumerating the features of all paths.
        """
        tree = self.required_inputs_for_property(end_property)
        return tree.get_paths_from(start_property)

    def get_degree_of_separation(self, start_property: Union[str, Symbol], end_property: Union[str, Symbol]) -> int:
        """
        Returns the minimum number of models separating two properties.
        Returns 0 if the start_property and end_property are equal.
        Returns None if the start_property and end_properties are not connected.
        """
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
        Generates all combinatorially-unique sets of input dicts given
        a list of property names and a quantity pool
        Args:
            props ([str]): property names
            this_quantity_pool ({Symbol: Set(Quantity)}): quantities
                keyed by symbols
        Returns ([{str: Quantity}]):
            list of symbol strings mapped to Quantity values.
        """
        aggregated_symbols = []
        for prop in props:
            if prop not in this_quantity_pool.keys():
                return []
            aggregated_symbols.append(this_quantity_pool[prop])
        return product(*aggregated_symbols)

    def get_input_sets_for_model(self, model, fixed_quantity, quantity_pool):
        """
        Generates all of the valid input sets for a given model, a fixed
        quantity, and a quantity pool from which to draw remaining properties
        Args:
            model (Model): model for which to evaluate valid input sets
            fixed_quantity (Quantity): quantity which must be included
                in all input sets
            quantity_pool ({symbol: {Quantity}}): dict of quantity sets
                keyed by symbol from which to draw additional quantities
                for model inputs
        Returns:
            list of sets of input quantities for the model
        """
        evaluation_lists = [c for c in model.evaluation_list
                            if fixed_quantity.symbol in c]
        all_input_sets = []
        for elist in evaluation_lists:
            elist_without_fixed = elist.copy()
            elist_without_fixed.remove(fixed_quantity.symbol)
            input_sets = self.generate_input_sets(elist_without_fixed, quantity_pool)
            for input_set in input_sets:
                all_input_sets.append(list(input_set) + [fixed_quantity])
        return all_input_sets

    def generate_models_and_input_sets(self, new_quantities, quantity_pool):
        """
        Helper method to generate input sets for models
        Args:
            new_quantities ([Quantity]): list of new quantities from which
                to derive new input sets (these are "fixed" quantities
                in generate_input_sets_for_model)
            quantity_pool ({symbol: {Quantity}}): dict of quantity sets
                keyed by symbol from which to draw additional quantities
                for model inputs
        Returns:
            ([tuple]): list of tuples of models and their associated input
                sets, uses tuple so duplicate checking can be performed
        """
        models_and_input_sets = []
        for quantity in new_quantities:
            for model in self._input_to_model[quantity.symbol]:
                input_sets = self.get_input_sets_for_model(
                    model, quantity, quantity_pool)
                models_and_input_sets += [
                    tuple([model] + sorted(list(input_set), key=lambda x: (x.symbol.name, x.value)))
                          for input_set in input_sets]
        # Filter for duplicates
        return set(models_and_input_sets)

    def derive_quantities(self, new_quantities, quantity_pool=None,
                          allow_model_failure=True, executor=None):
        """
        Algorithm for expanding quantity pool
        Args:
            new_quantities ([Quantity]): list of quantities which to
                consider as new inputs to models
            quantity_pool ({symbol: {Quantity}}): dict of quantity sets
                keyed by symbol from which to draw additional quantities
                for model inputs
        Returns:
            additional_quantities ([Quantity]): new derived quantities
            quantity_pool ({symbol: {Quantity}}): augmented version of
                quantity pool
        """
        # Update quantity pool
        quantity_pool = quantity_pool or defaultdict(set)
        for quantity in new_quantities:
            quantity_pool[quantity.symbol].add(quantity)

        run_serial = False
        if executor is None:
            run_serial = True

        # Generate all of the models and input sets to be evaluated
        logger.info("Generating models and input sets for %s", new_quantities)
        models_and_input_sets = self.generate_models_and_input_sets(
            new_quantities, quantity_pool)
        # print("Found {} input sets".format(len(models_and_input_sets)))
        # Evaluate model for each input set and add new valid quantities
        if run_serial:
            added_quantities = Graph._run_serial(models_and_input_sets,
                                                 allow_model_failure=allow_model_failure)
        else:
            added_quantities = Graph._run_parallel(executor, self._n_workers, models_and_input_sets,
                                                   allow_model_failure=allow_model_failure)

        return added_quantities, quantity_pool

    @staticmethod
    def _run_serial(models_and_input_sets, allow_model_failure=True):
        outputs = []
        # print("Processing {} input sets serially".format(len(models_and_input_sets)))
        for model_and_input_set in models_and_input_sets:
            try:
                out = Graph._evaluate_model(model_and_input_set,
                                            allow_failure=allow_model_failure,
                                            serialized=False)
                outputs.extend(out)
            except Exception as ex:
                if not allow_model_failure:
                    raise ex
        # print("Found {} new quantities".format(len(outputs)))
        return outputs

    @staticmethod
    def _run_parallel(executor, n_workers, models_and_input_sets, allow_model_failure=True):
        serialized_input_sets = []
        # print("Serializing {} input sets".format(len(models_and_input_sets)))
        for model_and_input_set in models_and_input_sets:
            model = model_and_input_set[0].name
            input_set = model_and_input_set[1:]
            serialized_inputs = [v.as_dict() for v in input_set]
            serialized_input_sets.append((model, *serialized_inputs))
        # print("Processing {} serialized input sets".format(len(serialized_input_sets)))
        func = partial(Graph._evaluate_model, allow_failure=allow_model_failure, serialized=True)
        chunk_size = int(len(serialized_input_sets) / n_workers) + 1
        # print("Chunk size: {}".format(chunk_size))
        results = executor.map(func, serialized_input_sets, chunksize=chunk_size)
        outputs = []
        for new_qs in results:
            for d in new_qs:
                if isinstance(d, Exception) and not allow_model_failure:
                    raise d
                outputs.append(QuantityFactory.from_dict(d))

        # print("Found {} new quantities".format(len(outputs)))
        return outputs

    @staticmethod
    def _evaluate_model(model_and_input_set, allow_failure=True, serialized=True):
        serialized_model = model_and_input_set[0]
        serialized_inputs = model_and_input_set[1:]
        if serialized:
            model = DEFAULT_MODEL_DICT[serialized_model]
            inputs = [QuantityFactory.from_dict(v) for v in serialized_inputs]
        else:
            model = serialized_model
            inputs = serialized_inputs
        input_dict = {q.symbol: q for q in inputs}
        logger.info('Evaluating %s with input %s', model, input_dict)
        # print('Evaluating {} with input {}'.format(model.name, input_dict))
        # sys.stdout.flush()

        try:
            with Timer(model.name):
                result = model.evaluate(input_dict,
                                        allow_failure=allow_failure)
        except Exception as ex:
            # If we get an exception, then allow_failure should be False
            # or we got some other exception we need to know about
            out = [ex]
            return out
        # TODO: Maybe provenance should be done in evaluate?

        success = result.pop('successful')
        if success:
            # print("Model {} evaluation successful".format(model.name))
            if serialized:
                out = [v.as_dict() for v in result.values() if not v.is_cyclic()]
            else:
                out = [v for v in result.values() if not v.is_cyclic()]
        else:
            # print("Model {} evaluation UNsuccessful".format(model.name))
            logger.info("Model evaluation unsuccessful %s",
                        result['message'])
            out = []
        return out

    def evaluate(self, material, allow_model_failure=True, parallel=False, max_workers=None):
        """
        Given a Material object as input, creates a new Material object
        to include all derivable properties.  Optional argument limits the
        scope of which models or properties are tested. Returns a
        reference to the new, augmented Material object.
        Args:
            material (Material): which material's properties will be expanded.
            allow_model_failure (Bool): whether to continue with graph evaluation
            if a model fails.
        Returns:
            (Material) reference to the newly derived material object.
        """
        logger.debug("Beginning evaluation")

        # Generate initial quantity set and pool
        new_quantities = material.get_quantities()
        quantity_pool = None

        if parallel:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
            if max_workers is None:
                self._n_workers = cpu_count()
            else:
                self._n_workers = max_workers
        else:
            executor = None

        # clear existing model evaluation statistics
        # clear()

        # Derive new Quantities
        # Loop util no new Quantity objects are derived.
        logger.debug("Beginning main loop with quantities %s", new_quantities)
        while new_quantities:
            new_quantities, quantity_pool = self.derive_quantities(
                new_quantities, quantity_pool,
                allow_model_failure=allow_model_failure,
                executor=executor)

        # store model evaluation statistics
        self._timings = timings

        new_material = Material()
        new_material._symbol_to_quantity = quantity_pool
        return new_material

    def super_evaluate(self, material, allow_model_failure=True):
        """
        Given a SuperMaterial object as input, creates a new SuperMaterial
        object to include all derivable properties.  Returns a reference to
        the new, augmented SuperMaterial object.
        Args:
            material (SuperMaterial): material for which properties
                will be expanded.
        Returns:
            (Material) reference to the newly derived material object.
        """

        if not isinstance(material, CompositeMaterial):
            raise Exception("material provided is not a SuperMaterial: " + str(type(material)))

        # Evaluate material's sub-materials
        evaluated_materials = list()
        for m in material.materials:
            logger.debug("Evaluating sub-material: " + str(id(m)))
            if isinstance(m, CompositeMaterial):
                evaluated_materials.append(self.super_evaluate(m))
            else:
                evaluated_materials.append(self.evaluate(m))

        # Run all SuperModels in the graph on this SuperMaterial if
        # a material mapping can be established.  Store any derived quantities.
        all_quantities = defaultdict(set)
        for (k, v) in material._symbol_to_quantity:
            all_quantities[k].add(v)

        to_return = CompositeMaterial(evaluated_materials)
        to_return._symbol_to_quantity = all_quantities

        logger.debug("Evaluating SuperMaterial")

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
                    symbol_list.append(CompositeModel.get_symbol(item))
                for i in range(0,len(mat_list)):
                    if mat_list[i] == None:     # Draw symbol from the CompositeMaterial
                        mat = to_return
                    else:
                        mat = mat_mapping[mat_list[i]]
                    for q in mat._symbol_to_quantity[symbol_list[i]]:
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
                        evaluate_set, allow_failure=allow_model_failure)
                    success = output.pop('successful')
                    if not success:
                        logger.debug("\t\t\tInput set failed -- did not produce a successful output.")
                        continue

                    # input_set led to output from the Model -- add output to the SuperMaterial

                    logger.debug("\t\t\tInput set produced successful output.")
                    for symbol, quantity in output.items():
                        st = self._symbol_types.get(symbol)
                        if not st:
                            raise ValueError(
                                "Symbol type {} not found".format(symbol))
                        q = QuantityFactory.create_quantity(st, quantity)
                        to_return._symbol_to_quantity[st].add(q)
                        logger.debug("\t\t\tNew output: " + str(q))

        # Evaluate the SuperMaterial's quantities and return the result.
        mappings = self.evaluate(to_return)._symbol_to_quantity
        to_return._symbol_to_quantity = mappings
        return to_return

    @property
    def evaluation_statistics(self):
        """
        :return: A Pandas DataFrame containing statistics on how
        many times each model was evaluated, average time per model,
        and the total time taken for that model.
        """

        rows = [{'Model Name': model,
                 'Total Evaluation Time /s': stats['total_elapsed'],
                 'Average Evaluation Time /s': stats['average_elapsed'],
                 'Number of Evaluations': stats['count']}
                for model, stats in self._timings.items()]

        return DataFrame(rows, columns=['Model Name',
                                        'Total Evaluation Time /s',
                                        'Number of Evaluations',
                                        'Average Evaluation Time /s'])
