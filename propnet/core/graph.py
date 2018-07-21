"""
Module containing classes and methods for graph functionality in Propnet code.
"""

from typing import *

import networkx as nx

from propnet.models import DEFAULT_MODELS
from propnet.symbols import DEFAULT_SYMBOLS

from propnet.core.quantity import Quantity
from propnet.core.materials import Material, SuperMaterial
import logging
from itertools import chain, product

logger = logging.getLogger("graph")


# TODO: reconsider polymorphism
class Graph(object):
    """
    Class containing methods for creating and interacting with a Property Network.

    The Property Network contains a set of Node namedtuples with connections stored as directed edges between
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

        _symbol_types ({str: Symbol}): data structure mapping Symbol name to Symbol object.
        _models ({str: Model}): data structure mapping Model name to Model object.
        _input_to_model ({Symbol: {Model}}): data structure mapping Symbol inputs to a set of corresponding Model
                                             objects that input that Symbol.
        _output_to_model ({Symbol: {Model}}): data structure mapping Symbol outputs to a set of corresponding Model
                                              objects that output that Symbol.

        _super_models ({str: SuperModel}): data structure mapping SuperModel name to SuperModel object.


    *** Dictionaries can be searched by supplying Symbol objects or Strings as to their names.

    """

    def __init__(self, models=None, super_models=None, symbol_types=None):
        """
        Creates a Graph instance
        """

        # set our defaults if no models/symbol types supplied
        if models is None:
            defaults = dict()
            for (k, v) in DEFAULT_MODELS.items():
                defaults[k] = v()
            models = defaults
        symbol_types = symbol_types or DEFAULT_SYMBOLS

        # create the graph
        self._symbol_types = dict()
        self._models = dict()
        self._super_models = dict()
        self._input_to_model = DefaultDict(set)
        self._output_to_model = DefaultDict(set)

        if symbol_types:
            self.update_symbol_types(symbol_types)

        if models:
            self.update_models(models)

    def __str__(self):
        """
        Returns a full summary of the graph in terms of the SymbolTypes, Symbols, Materials, and Models
        that it contains. Connections are shown as nesting within the printout.

        Returns:
            (str) representation of this Graph object.
        """
        QUANTITY_LENGTH_CAP = 35
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
            for d in model.type_connections:
                try:
                    symbol_inputs = [self._symbol_types[symb_str] for symb_str in d['inputs']]
                    symbol_outputs = [self._symbol_types[symb_str] for symb_str in d['outputs']]
                    for symbol in symbol_inputs:
                        self._input_to_model[symbol].add(model)
                    for symbol in symbol_outputs:
                        self._output_to_model[symbol].add(model)
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

    def update_super_models(self, super_models):
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
            self._super_models[model.name] = model
            added[model.name] = model
            for d in model.type_connections:
                try:
                    symbol_inputs = [self._symbol_types[symb_str] for symb_str in d['inputs']]
                    symbol_outputs = [self._symbol_types[symb_str] for symb_str in d['outputs']]
                except KeyError as e:
                    self.remove_super_models(added)
                    raise KeyError('Attempted to add a model to the property network with an unrecognized Symbol.\
                                            Add {} Symbol to the property network before adding this model.'.format(e))

    def remove_super_models(self, super_models):
        """
        Remove user-defined models from the Graph.
        Args:
            super_models (dict<str, SuperModel>): Instances of the SuperModel class
        Returns:
            None
        """
        for model in super_models.keys():
            if model not in self._super_models.keys():
                raise Exception("Attempted to remove a model not currently present in the graph.")
            del self._super_models[model]

    def get_super_models(self):
        """
        Getter method, returns a set of all model objects present on the graph.
        Returns:
            (set<Model>)
        """
        to_return = set()
        for model in self._super_models.values():
            to_return.add(model)
        return to_return

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
                for s in m.output_symbol_types:
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
                for d in m.type_connections:
                    has_inputs = True
                    for s in d['inputs']:
                        if s not in working:
                            has_inputs = False
                            break
                    if not has_inputs:
                        continue
                    # Check passed -- add model outputs to the available properties.
                    #              -- add any new models working with these newly available properties
                    for s in d['outputs']:
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

    @staticmethod
    def generate_input_sets(symbol_list, req_types, this_quantity_pool):
            """
            Generates all combinatorially-unique sets of input dictionaries.

            Args:
                symb_list ([str]): list of model symbols mapped to Symbol objects.
                req_types ([str]): list of Symbols that must be retrieved from the quantity_pool.
                this_quantity_pool ({Symbol: Set(Quantity)}): quantities keyed
                    by symbols

            Returns:
                ([{str: Quantity}]): list of symbol strings mapped to Quantity values.
            """
            if len(symbol_list) != len(req_types):
                raise Exception("Symbol and Type sets must be the same length.")
            aggregated_symbols = [this_quantity_pool[req_type]
                                  for req_type in req_types]
            input_set_lists = product(*aggregated_symbols)
            input_set_dicts = []
            for input_set_list in input_set_lists:
                input_set_dicts.append({
                    symbol: input_quantity for symbol, input_quantity
                    in zip(symbol_list, input_set_list)
                })
            return input_set_dicts

    def evaluate(self, material, property_type=None):
        """
        Given a Material object as input, creates a new Material object to include all derivable properties.
        Returns a reference to the new, augmented Material object.

        Optional argument limits the scope of which models or properties are tested.
            property_type parameter: produces output from models only if all input properties are in the list.

        Args:
            material (Material): which material's properties will be expanded.
            property_type (set<Symbol>): optional limit on which Symbols will be considered as input.
        Returns:
            (Material) reference to the newly derived material object.
        """

        # Determine which Quantity objects are up for evaluation.
        # Generate the necessary initial data-structures.

        logger.debug("Beginning evaluation")

        quantity_pool = DefaultDict(set)   # Dict<Symbol, set<Quantity>>, available Quantity objects.
        plug_in_dict = DefaultDict(set)    # Dict<Quantity, set<Model>>, where the Quantities have been plugged in.
        output_dict = DefaultDict(set)     # Dict<Quantity, set<Model>>, where the Quantities have been generated.
        candidate_models = set()           # set<Model>, which could generate additional outputs.

        logger.debug("Refining input set, setting up candidate_models.")

        for qs in material._symbol_to_quantity.values():
            for quantity in qs:
                if property_type is None or quantity.symbol_type in property_type:
                    quantity_pool[quantity.symbol].add(quantity)

        for symbol in quantity_pool.keys():
            for m in self._input_to_model[symbol]:
                candidate_models.add(m)

        logger.debug("Finished refining input set.")
        logger.debug("Quantity pool contains {}".format(quantity_pool))
        logger.debug("Beginning main loop.")

        # Derive new Quantities
        # Loop util no new Quantity objects are derived.

        new_models = set()
        continue_loop = True

        while continue_loop:
            continue_loop = False

            # Clean up after last loop.

            for model in new_models:
                candidate_models.add(model)
            new_models = set()

            logger.debug("Checking if model inputs are supplied.")

            for model in candidate_models:

                logger.debug("Checking model {}".format(model.title))
                logger.debug("Quantity pool contains {} quantities:".format(
                    len(list(chain.from_iterable(quantity_pool.values())))))

                inputs = model.gen_evaluation_lists()

                for l in inputs:

                    logger.debug("\tGenerating input sets for: " + str(l))

                    input_sets = self.generate_input_sets(l[0], l[1], quantity_pool)

                    for input_set in input_sets:

                        logger.debug("\t\tEvaluating input set: " + str(input_set))

                        override = False
                        can_evaluate = False

                        # Check if input_set can be evaluated --
                        #       input_set has never been seen before by the model
                        #       input_set contains no values that were previously derived from the model
                        #       input_set must pass the necessary model constraints

                        for q in input_set.values():
                            if model in output_dict[q]:
                                override = True
                                break
                        if override:
                            logger.debug("\t\t\tInput set failed -- input previously derived from the model.")
                            continue
                        for q in input_set.values():
                            if model not in plug_in_dict[q]:
                                can_evaluate = True
                                break
                        if not can_evaluate:
                            logger.debug("\t\t\tInput set failed -- input set previously plugged in to the model.")
                            continue
                        if not model.check_constraints(input_set):
                            logger.debug("\t\t\tInput set failed -- did not pass model constraints.")
                            continue

                        # Try to evaluate input_set:

                        evaluate_set = dict()
                        for symbol, quantity in input_set.items():
                            evaluate_set[symbol] = quantity.value
                        output = model.evaluate(evaluate_set)
                        if not output['successful']:
                            logger.debug("\t\t\tInput set failed -- did not produce a successful output.")
                            continue

                        # input_set led to output from the Model -- gather output
                        #                       -- add output to the graph
                        #                       -- add additional candidate models

                        logger.debug("\t\t\tInput set produced successful output.")
                        continue_loop = True
                        for symbol, quantity in output.items():
                            st = self._symbol_types.get(
                                model.symbol_mapping.get(symbol))
                            if not st:
                                logger.debug("\t\t\tUnrecognized symbol_type in the output: " + str(symbol))
                                continue
                            for m in self._input_to_model[st]:
                                new_models.add(m)
                            q = Quantity(st, quantity)
                            quantity_pool[st].add(q)
                            output_dict[q].add(model)
                            logger.debug("\t\t\tNew output: " + str(q))

                            # Derive the chain of all models that were required
                            # to get to the new quantity

                            for input_quantity in input_set.values():
                                for link in output_dict[input_quantity]:
                                    output_dict[q].add(link)

                    # Store all input sets to avoid duplicate evaluation in the future.
                    for input_set in input_sets:
                        for quantity in input_set.values():
                            plug_in_dict[quantity].add(model)

        toReturn = Material()
        toReturn._symbol_to_quantity = quantity_pool
        return toReturn

    def super_evaluate(self, material, property_type=None):
        """
        Given a SuperMaterial object as input, creates a new SuperMaterial object to include all derivable properties.
        Returns a reference to the new, augmented SuperMaterial object.

        Optional argument limits the scope of which models or properties are tested.
            property_type parameter: produces output from models only if all input properties are in the list.

        Args:
            material (SuperMaterial): which material's properties will be expanded.
            property_type (set<Symbol>): optional limit on which Symbols will be considered as input.
        Returns:
            (Material) reference to the newly derived material object.
        """

        if not isinstance(material, SuperMaterial):
            raise Exception("material provided is not a SuperMaterial: " + str(type(material)))

        # Evaluate material's sub-materials

        evaluated_materials = set()
        for m in material.materials:
            logger.debug("Evaluating sub-material: " + str(id(m)))
            if isinstance(m, SuperMaterial):
                evaluated_materials.add(self.super_evaluate(m, property_type=property_type))
            else:
                evaluated_materials.add(self.evaluate(m, property_type=property_type))

        # Run all SuperModels in the graph on this SuperMaterial if a material mapping can be established.
        # Store any derived quantities.

        all_quantities = DefaultDict(set)
        for (k,v) in material._symbol_to_quantity:
            all_quantities[k].add(v)

        logger.debug("Evaluating SuperMaterial")

        for model in self._super_models.values():

            logger.debug("\tEvaluating Model: " + m.name)

            # Check to see if a valid material_mapping can be established.

            mat_mapping = m.material_mapping(material)
            if not mat_mapping:
                logger.debug("\t\tEvaluation failed: no valid material mapping.")
                continue

            # Go through input sets

            inputs = model.gen_evaluation_lists()
            for l in inputs:

                logger.debug("\t\tGenerating input sets for: " + str(l))

                # Create a quantity pool from the appropriate materials.
                # Modify inputs for use in generate_input_sets

                temp_pool = DefaultDict(set)
                new_spec = []
                for pair in l[1]:
                    for p_material in pair[1]:
                        for q in p_material._symbol_to_quantity[pair[0]]:
                            temp_pool[q._symbol_type].add(q)
                    new_spec.append(pair[0])
                input_sets = self.generate_input_sets(l[0], new_spec, temp_pool)

                for input_set in input_sets:

                    logger.debug("\t\t\tEvaluating input set: " + str(input_set))

                    # Check if input_set can be evaluated -- input_set must pass the necessary model constraints

                    if not model.check_constraints(input_set):
                        logger.debug("\t\t\tInput set failed -- did not pass model constraints.")
                        continue

                    # Try to evaluate input_set:

                    evaluate_set = dict()
                    for symbol, quantity in input_set.items():
                        evaluate_set[symbol] = quantity.value
                    output = model.evaluate(evaluate_set)
                    if not output['successful']:
                        logger.debug("\t\t\tInput set failed -- did not produce a successful output.")
                        continue

                    # input_set led to output from the Model -- add output to the SuperMaterial

                    logger.debug("\t\t\tInput set produced successful output.")

                    for symbol, quantity in output.items():
                        st = self._symbol_types.get(
                            model.symbol_mapping.get(symbol))
                        if not st:
                            logger.debug("\t\t\tUnrecognized symbol_type in the output: " + str(symbol))
                            continue
                        q = Quantity(st, quantity)
                        all_quantities[st].add(q)
                        logger.debug("\t\t\tNew output: " + str(q))

        # Evaluate the SuperMaterial's quantities and return the result.
        to_return = SuperMaterial(evaluated_materials)
        to_return._symbol_to_quantity = all_quantities
        return self.evaluate(to_return)


class SymbolPath(object):
    """
    Utility class to store elements of a Symbol path through
    various inputs and outputs.
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


class SymbolTree(object):
    """
    Wrapper around TreeElement data structure for export from
    the method, encapsulating functionality.
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


class TreeElement(object):
    """
    Tree-like data structure for representing property
    relationship paths.
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
