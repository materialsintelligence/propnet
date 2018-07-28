"""
Module containing classes and methods for graph functionality in Propnet code.
"""

from typing import *

import networkx as nx

from propnet.models import DEFAULT_MODEL_DICT
from propnet.symbols import DEFAULT_SYMBOLS

from propnet.core.quantity import Quantity
import logging
from itertools import chain

logger = logging.getLogger("graph")



class Graph:
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
        _models ({str: Symbol}): data structure mapping Model name to
            Model object.
        _materials ({Material}): data structure storing the set of all
            materials present on the graph.
        _input_to_model ({Symbol: {Model}}): data structure mapping
            Symbol inputs to a set of corresponding Model objects that
            input that Symbol.
        _output_to_model ({Symbol: {Model}}): data structure mapping
            Symbol outputs to a set of corresponding Model objects that
            output that Symbol.
        _symbol_to_quantity ({Symbol: {Quantity}}): data structure
            mapping Symbols to a list of corresponding Quantity objects
            of that type.

    *** Dictionaries can be searched by supplying Symbol objects or
        Strings as to their names.

    """

    def __init__(self, materials=None, models=None, symbol_types=None):
        """
        Creates a Graph instance
        """

        # set our defaults if no models/symbol types supplied
        defaults = models or DEFAULT_MODEL_DICT
        symbol_types = symbol_types or DEFAULT_SYMBOLS

        # create the graph
        self._symbol_types = dict()
        self._models = dict()
        self._materials = set()
        self._input_to_model = DefaultDict(set)
        self._output_to_model = DefaultDict(set)
        self._symbol_to_quantity = DefaultDict(set)

        if symbol_types:
            self.update_symbol_types(symbol_types)

        if models:
            self.update_models(models)

        if materials:
            for material in materials:
                self.add_material(material)

    def __str__(self):
        """
        Returns a full summary of the graph in terms of the SymbolTypes,
        Symbols, Materials, and Models that it contains. Connections are
        shown as nesting within the printout.

        Returns:
            (str) representation of this Graph object.
        """
        QUANTITY_LENGTH_CAP = 35
        summary = ["Propnet Printout", ""]
        summary += ["Properties"]
        for property in self._symbol_types.keys():
            summary += ["\t" + property]
            if property not in self._symbol_to_quantity.keys():
                continue
            for quantity in self._symbol_to_quantity[self._symbol_types[property]]:
                qs = str(quantity)
                if "\n" in qs or len(qs) > QUANTITY_LENGTH_CAP:
                    qs = "..."
                summary += ["\t\tValue: " + qs]
        summary += [""]
        summary += ["Models"]
        for model in self._models.keys():
            summary += ["\t" + model]
        summary += [""]
        return "\n".join(summary)

    def update_symbol_types(self, symbol_types):
        """
        Add / redefine user-defined symbol types to the graph.
        If the input, symbol_types, includes keys in self._symbol_types, they are redefined.
        Args:
            symbol_types (dict<str, Symbol>): {name:Symbol}
        Returns:
            None
        """
        for (k, v) in symbol_types.items():
            self._symbol_types[k] = v

    def remove_symbol_types(self, symbol_types):
        """
        Removes user-defined Symbol objects to the Graph.
        Removes any models that input or output this Symbol because they are no longer defined
        without the given symbol_types.
        Args:
            symbol_types (dict<str, Symbol>): {name:Symbol}
        Returns:
            None
        """
        models_to_remove = {}
        for symbol in symbol_types.keys():
            if symbol not in self._symbol_types.keys():
                raise Exception("Trying to remove a symbol that is not currently defined.")
            if symbol_types[symbol] != self._symbol_types[symbol]:
                raise Exception("Trying to remove a symbol that is not currently defined.")
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
        Getter method, returns a set of all Symbol objects present on the graph.
        Returns:
            (set<Symbol>)
        """
        to_return = set()
        for s in self._symbol_types.values():
            to_return.add(s)
        return to_return

    def update_models(self, models):
        """
        Add / redefine user-defined models to the graph.
        If the input, models, includes keys in self._models, they are redefined.
        The addition of a model may fail if appropriate Symbol objects are not already on the graph.
        If any addition operation fails, the entire update is aborted.
        Args:
            models (dict<str, Model>): Instances of the model class (subclasses AbstractModel)
        Returns:
            None
        """
        added = {}
        for model in models.values():
            self._models[model.name] = model
            added[model.name] = model
            try:
                for input_set in model.input_sets:
                    for property in input_set:
                        self._input_to_model[property].add(model)
                for output_set in model.output_sets:
                    for property in output_set:
                        self._output_to_model[property].add(model)
            except KeyError as e:
                self.remove_models(added)
                raise KeyError('Attempted to add a model to the property network with an unrecognized Symbol.\
                                Add {} Symbol to the property network before adding this model.'.format(e))

    def remove_models(self, models):
        """
        Remove user-defined models from the Graph.
        Args:
            models (dict<str, Model>): Instances of the model class
        Returns:
            None
        """
        for model in models.keys():
            if model not in self._models.keys():
                raise Exception("Attempted to remove a model not currently present in the graph.")
            del self._models[model]
        for s in self._input_to_model.values():
            for model in models.values():
                if model in s:
                    s.remove(model)
        for s in self._output_to_model.values():
            for model in models.values():
                if model in s:
                    s.remove(model)

    def get_models(self):
        """
        Getter method, returns a set of all model objects present on the graph.
        Returns:
            (set<Model>)
        """
        to_return = set()
        for model in self._models.values():
            to_return.add(model)
        return to_return

    def add_material(self, material):
        """
        Add a material and any of its associated properties to the Graph.
        Mutates the graph instance variable.
        Args:
            material (Material) Material whose information will be added to the graph.
        Returns:
            void
        """
        if material in self._materials:
            raise Exception("Material has already been added to the graph.")
        self._materials.add(material)
        material.parent = self
        for qs in material._symbol_to_quantity.values():
            for q in qs:
                self._add_quantity(q)

    def remove_material(self, material):
        """
        Removes a material and any of its associated properties from the Graph.
        Mutates the graph instance variable.

        Args:
            material (Material) Material whose information will be removed from the graph.
        Returns:
            void
        """
        if material not in self._materials:
            raise Exception("Trying to remove material that is not part of the graph.")
        self._materials.remove(material)
        for qs in list(material._symbol_to_quantity.values()):
            for q in qs:
                self._remove_quantity(q)
        material.parent = None

    def get_materials(self):
        """
        Getter method returning all materials on the graph.
        Returns:
            (set<Material>)
        """
        return {m for m in self._materials}

    def _add_quantity(self, property):
        """
        PRIVATE METHOD!
        Adds a property to this graph. Properties should be added to Material objects ONLY.
        This method is called ONLY by Material objects to ensure consistency of data structures.
        Args:
            property (Quantity):
        Returns:
            None
        """
        if property.symbol.name not in self._symbol_types:
            raise KeyError("Attempted to add a Quantity to the graph for which no corresponding Symbol exists.\
                            Please add the appropriate Symbol to the property network and try again.")
        self._symbol_to_quantity[property.symbol].add(property)

    def _remove_quantity(self, property):
        """
        PRIVATE METHOD!
        Removes this property from the graph. Properties should be removed from Material objects ONLY.
        This method is called ONLY by Material objects to ensure consistency of data structures.
        Args:
            property (Quantity): the property to be removed

        Returns:
            None
        """
        if property.symbol.name not in self._symbol_types:
            raise Exception("Attempted to remove a quantity not part of the graph.")
        self._symbol_to_quantity[property.symbol].remove(property)
        if len(self._symbol_to_quantity[property.symbol]) == 0:
            del self._symbol_to_quantity[property.symbol]

    @property
    def graph(self):
        """
        Generates a networkX data structure representing the property network and returns
        this object.
        Returns:
            (networkX.multidigraph)
        """
        graph = nx.MultiDiGraph()

        # Create the abstract graph.
        for symbol in self._input_to_model:
            for model in self._input_to_model[symbol]:
                graph.add_edge(symbol, model)
        for symbol in self._output_to_model:
            for model in self._output_to_model[symbol]:
                graph.add_edge(model, symbol)

        # Add the concrete graph.
        for symbol in self._symbol_to_quantity:
            for quantity in self._symbol_to_quantity[symbol]:
                for material in quantity._material:
                    graph.add_edge(material, quantity)
                graph.add_edge(quantity, symbol)

        # Add orphan nodes
        for symbol in self._symbol_types:
            if not symbol in graph.nodes:
                graph.add_node(symbol)

        return graph

    def calculable_properties(self, property_type_set):
        """
        Given a set of Symbol objects, returns all new Symbol objects that may be calculable from the inputs.
        Resulting set contains only those new Symbol objects derivable.
        The result should be used with caution:
            1) Models may not produce an output if their input conditions are not met.
            2) Models may require more than one Quantity of a given Symbol type to generate an output.

        Args:
            property_type_set (set<Symbol>): the set of Symbol objects taken as starting properties.
        Returns:
            ((set<Symbol>, set<Model>)) the set of all Symbol objects that can be derived from the property_type_set,
                                        the set of all Model objects that are used in deriving the new Symbol objects.
        """
        # Set of theoretically derivable properties.
        derivable = set()

        # Set of theoretically available properties.
        working = set()
        for p in property_type_set:
            working.add(p)

        # Set of all models that could produce output.
        all_models = set()
        c_models = set()
        for p in property_type_set:
            for m in self._input_to_model[p]:
                all_models.add(m)
                c_models.add(m)

        to_add = set()
        to_remove = set()

        has_changed = True

        # TODO: revisit this, looks a bit too complicated
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
            for m in c_models:
                # Check if model can add a new Symbols
                can_contribute = False
                for s in m.all_outputs:
                    if s not in working:
                        can_contribute = True
                        break
                if not can_contribute:
                    to_remove.add(m)
                    continue
                # Check if model has all constraint Symbols provided.
                has_inputs = True
                for s in m.type_constraint_symbols():
                    if s not in working:
                        has_inputs = False
                        break
                if not has_inputs:
                    continue
                # Check if any model input sets are met.
                for input_set in m.input_sets:
                    has_inputs = True
                    for s in input_set:
                        if s not in working:
                            has_inputs = False
                            break
                    if not has_inputs:
                        continue
                    # Check passed -- add model outputs to the available properties.
                    #              -- add any new models working with these newly available properties
                for output_set in m.output_sets:
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
        Determines all potential paths leading to a given symbol object. Answers the question:
            What sets of properties are required to calculate this given property?
        Paths are represented as a series of models and required input Symbol objects.
        Paths can be searched to determine specifically how to get from one property to another.

        Warning: Method indicates sets of Symbol objects required to calculate the property.
                 It does not indicate how many of each Symbol is required.
                 It does not guarantee that supplying Quantities of these types will result in a
                 new Symbol output as conditions / assumptions may not be met.

        Returns:
            SymbolTree
        """
        head = TreeElement(None, {property}, None, None)
        self._tree_builder(head)
        return SymbolTree(head)

    def _tree_builder(self, to_expand):
        """
        Recursive helper method to build a SymbolTree.
        Fills in the children of to_expand by all possible model substitutions.
        Args:
            to_expand: (TreeElement) element that will be expanded
        Returns:
            None
        """

        # Get set of symbols that no longer need to be replaced and symbls that are candidates for replacement.
        replaced_symbols = set()    # set of all symbols that have already been replaced.
                                    # equal to all parents' symbols minus to_expand's symbols.
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
        prev = DefaultDict(list)
        for symbol in candidate_symbols:
            c_models = self._output_to_model[symbol]
            for model in c_models:
                for d in model.type_connections:
                    can_continue = True
                    for input_symbol in d['inputs']:
                        if input_symbol in replaced_symbols:
                            can_continue = False
                            break
                    if not can_continue:
                        continue
                    s_inputs = set(d['inputs'] + model.type_constraint_symbols())
                    s_outputs = set(d['outputs'])
                    new_types = (to_expand.inputs - s_outputs)
                    new_types.update(s_inputs)
                    new_types = {self._symbol_types[x] for x in new_types}
                    if new_types in prev[model]:
                        continue
                    prev[model].append(new_types)
                    new_element = TreeElement(model, new_types, to_expand, None)
                    self._tree_builder(new_element)
                    outputs.append(new_element)

        # Add outputs to children and fill in their elements.
        to_expand.children = outputs

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
        a Quantity might be derived from a combination of materials in this case. Likewise existing Material nodes'
        graph instances will not be mutated in this case.

        Args:
            material (Material): optional limit on which material's properties will be expanded (default: all materials)
            property_type (list<Symbol>): optional limit on which Symbols will be considered as input.
        Returns:
            void
        """

        # Determine which Quantity objects are up for evaluation.
        # Generate the necessary initial datastructures.
        logger.debug("Beginning evaluation")
        quantity_pool = DefaultDict(set)   # Dict<Symbol, set<Quantity>>, available Quantity objects.
        plug_in_dict = DefaultDict(set)    # Dict<Quantity, set<Model>>, where the Quantities have been plugged in.
        output_dict = DefaultDict(set)     # Dict<Quantity, set<Model>>, where the Quantities have been generated.
        candidate_models = set()           # set<Model>, which could generate additional outputs.

        logger.debug("Refining input set")
        for qs in self._symbol_to_quantity.values():
            for quantity in qs:
                if (material is None or material in quantity._material) and \
                        (property_type is None or quantity.symbol_type in property_type):
                    quantity_pool[quantity.symbol].add(quantity)

        for symbol in quantity_pool.keys():
            for m in self._input_to_model[symbol]:
                candidate_models.add(m)

        # Define helper closures for later use
        # TODO: I think these might be rewritten with itertools - montoyjh

        def gen_input_sets(symb_list, req_types, this_quantity_pool):
            """
            Generates all combinatorially-unique sets of input dictionaries.
            Args:
                symb_list (list<str>): list of model symbols mapped to Symbol objects.
                req_types (list<str>): list of Symbols that must be retrieved from the quantity_pool.
            Returns:
                (list<dict<str, Quantity>>): list of symbol strings mapped to Quantity values.
            """
            if len(symb_list) != len(req_types):
                raise Exception("Symbol and Type sets must be the same length.")
            type_mapping = DefaultDict(list)
            for i in range(0, len(symb_list)):
                # Create mapping Symbol to list<Model.Symbol.name>
                type_mapping[req_types[i]].append(symb_list[i])
            return gen_quantity_combos(type_mapping, this_quantity_pool)

        def gen_quantity_combos(type_mapping, this_quantity_pool):
            """
            Generates all combinatorially-unique sets of input dictionaries.
            """
            if len(type_mapping) == 0:
                return []
            key = type_mapping.__iter__().__next__()
            val = type_mapping[key]
            opt = list(this_quantity_pool[key])
            if len(val) > len(opt):
                return []
            if len(type_mapping) == 1:
                return gen_dict_combos(val, opt)
            elif len(type_mapping) > 1:
                next = gen_dict_combos(val, opt)
                del type_mapping[key]
                remaining = gen_quantity_combos(type_mapping, this_quantity_pool)
                to_return = [None]*len(next)*len(remaining)
                for i in range(0, len(next)):
                    for j in range(0, len(remaining)):
                        building = dict()
                        for (k, v) in next[i].items():
                            building[k] = v
                        for (k, v) in remaining[j].items():
                            building[k] = v
                        to_return[i*len(remaining) + j] = building
                return to_return

        def gen_dict_combos(val, opt):
            """
            Generates all combinatorial sets of mappings from Model symbol to Quantity
            """
            if len(val) == 1:
                to_return = [None]*len(opt)
                for i in range(0, len(opt)):
                    to_return[i] = {val[0]: opt[i]}
                return to_return
            else:
                first = gen_dict_combos([val[0]], opt)
                nexts = [None]*len(first)
                for i in range(0, len(first)):
                    next_val = val[1:len(val)]
                    next_opt = opt[0:i] + opt[(i+1):len(opt)]
                    p = 0
                    for j in range(0, len(opt)):
                        if i == j:
                            continue
                        next_opt[p] = opt[j]
                        p += 1
                    nexts[i] = gen_dict_combos(next_val, next_opt)
                to_return = [None]*(len(nexts)*len(nexts[0]))
                p = 0
                for i in range(0, len(nexts)):
                    for j in range(0, len(nexts[i])):
                        adding = dict()
                        for (k, v) in first[i].items():
                            adding[k] = v
                        for (k, v) in nexts[i][j].items():
                            adding[k] = v
                        to_return[p] = adding
                        p += 1
                return to_return

        # Derive new Quantities
        # Loop util no new Quantity objects are derived.

        add_set = set()

        continue_loop = True
        logger.debug("Beginning main loop")
        logger.debug("Quantity pool contains {}".format(quantity_pool))
        while continue_loop:
            continue_loop = False
            # Check if model inputs are supplied.
            logger.debug("Checking if model inputs are supplied")
            for model in add_set:
                candidate_models.add(model)
            add_set = set()
            for model in candidate_models:
                logger.debug("Evaluating model {}".format(model.title))
                logger.debug("Quantity pool contains {} quantities:".format(
                    len(list(chain.from_iterable(quantity_pool.values())))))
                inputs = model.evaluation_list
                for l in inputs:
                    logger.debug("Generating input sets")
                    input_sets = gen_input_sets(l[0], l[1], quantity_pool)
                    for input_set in input_sets:
                        override = False
                        can_evaluate = False
                        for q in input_set.values():
                            if model in output_dict[q]:
                                override = True
                                break
                            if model not in plug_in_dict[q] and model not in output_dict[q]:
                                can_evaluate = True
                                break
                        if override or not can_evaluate:
                            continue
                        if not model.check_constraints(input_set):
                            continue
                        mats = set()
                        for value in input_set.values():
                            for mat in value._material:
                                mats.add(mat)
                        evaluate_set = dict()
                        for (k, v) in input_set.items():
                            evaluate_set[k] = v.value
                        output = model.evaluate(evaluate_set)
                        if not output['successful']:
                            continue
                        # Model produced output -- gather output
                        #                       -- add output to the graph
                        #                       -- add additional candidate models
                        continue_loop = True
                        for (k, v) in output.items():
                            st = self._symbol_types.get(model.symbol_mapping.get(k))
                            if not st:
                                continue
                            for m in self._input_to_model[st]:
                                add_set.add(m)
                            q = Quantity(st, v, set())
                            for mat in mats:
                                mat.add_quantity(q)
                            quantity_pool[st].add(q)
                            output_dict[q].add(model)
                            for input_quantity in input_set.values():
                                for link in output_dict[input_quantity]:
                                    output_dict[q].add(link)
                    for input_set in input_sets:
                        for value in input_set.values():
                            plug_in_dict[value].add(model)


class SymbolPath:
    """
    Utility class to store elements of a Symbol path through various inputs and outputs.
    """

    __slots__ = ['symbol_set', 'model_path']

    def __init__(self, symbol_set, model_path):
        """
        Args:
            symbol_set: (set<Symbol>) set of all inputs required to complete the path
            model_path: (list<Model>) list of models, in order, required to complete the path
        """
        self.symbol_set = symbol_set
        self.model_path = model_path

    def __eq__(self, other):
        if not isinstance(other, SymbolPath):
            return False
        if not self.symbol_set == other.symbol_set:
            return False
        if not self.model_path == other.model_path:
            return False
        return True


class SymbolTree:
    """
    Wrapper around TreeElement data structure for export from the method, encapsulating functionality.
    """

    __slots__ = ['head']

    def __init__(self, head):
        """
        Args:
            head: (TreeElement) head of the tree.
        """
        self.head = head

    def get_paths_from(self, symbol):
        """
        Gets all paths from input to symbol
        Args:
            symbol: (Symbol) we are searching for paths from this symbol to head.
        Returns:
            (list<SymbolPath>)
        """
        to_return = []
        visitation_queue = [self.head]
        while len(visitation_queue) != 0:
            visiting = visitation_queue.pop(0)
            for elem in visiting.children:
                visitation_queue.append(elem)
            if symbol in visiting.inputs:
                v = visiting
                model_trail = []
                while v.parent is not None:
                    model_trail.append(v.m)
                    v = v.parent
                to_return.append(SymbolPath(visiting.inputs, model_trail))
        return to_return


class TreeElement:
    """
    Tree-like data structure for representing property relationship paths.
    """

    __slots__ = ['m', 'inputs', 'parent', 'children']

    def __init__(self, m, inputs, parent, children):
        """
        Args:
            m: (Model) model outputting the parent from children inputs
            inputs: (set<Symbol>) Symbol inputs required to produce the parent
            parent: (TreeElement)
            children: (list<TreeElement>) all PathElements derivable from this one
        """
        self.m = m
        self.inputs = inputs
        self.parent = parent
        self.children = children
