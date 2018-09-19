"""
Module containing classes and methods for graph functionality in Propnet code.
"""

import logging
from collections import defaultdict
from itertools import chain, product

import networkx as nx

from propnet.core.materials import CompositeMaterial
from propnet.core.materials import Material
from propnet.core.models import CompositeModel
from propnet.core.quantity import Quantity
from propnet.core.utils import ProvenanceElement, SymbolTree, TreeElement
from propnet.models import COMPOSITE_MODEL_DICT
from propnet.models import DEFAULT_MODEL_DICT
from propnet.symbols import DEFAULT_SYMBOLS

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
            input that Symbol.
        _output_to_model ({Symbol: {Model}}): data structure mapping
            Symbol outputs to a set of corresponding Model objects that
            output that Symbol.

    *** Dictionaries can be searched by supplying Symbol objects or
        Strings as to their names.

    """

    def __init__(self, models=None, composite_models=None, symbol_types=None):
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
                for input_set in model.input_sets:
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

    def get_models(self):
        """
        Getter method, returns a set of all model objects present
        on the graph.

        Returns ({Model}):
            set of models in the graph
        """
        to_return = set()
        for model in self._models.values():
            to_return.add(model)
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

    # TODO: deprecate this and use web app
    @property
    def graph(self):
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

        # TODO: revisit necessity of this
        # # Add the concrete graph.
        # for symbol in self._symbol_to_quantity:
        #     for quantity in self._symbol_to_quantity[symbol]:
        #         for material in quantity._material:
        #             graph.add_edge(material, quantity)
        #         graph.add_edge(quantity, symbol)

        # Add orphan nodes
        for symbol in self._symbol_types.values():
            if symbol not in graph.nodes:
                graph.add_node(symbol)

        return graph

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

    def _tree_builder(self, to_expand):
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
        # TODO: this also might be too complicated, marking for refactor
        for symbol in candidate_symbols:
            c_models = self._output_to_model[symbol]
            for model in c_models:
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
    def generate_input_sets(props, this_quantity_pool):
        """
        Generates all combinatorially-unique sets of input dicts.

        Args:
            properties ([str]): property names
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
        input_set_lists = product(*aggregated_symbols)
        input_set_dicts = []
        for input_set_list in input_set_lists:
            input_set_dicts.append({
                symbol: input_quantity for symbol, input_quantity
                in zip(props, input_set_list)
            })
        return input_set_dicts

    def evaluate(self, material, property_type=None):
        """
        Given a Material object as input, creates a new Material object
        to include all derivable properties.  Optional argument limits the
        scope of which models or properties are tested. Returns a
        reference to the new, augmented Material object.

        Args:
            material (Material): which material's properties will be expanded.
            property_type ({Symbol}): optional limit on which Symbols
                will be considered as input.
        Returns:
            (Material) reference to the newly derived material object.
        """

        # Determine which Quantity objects are up for evaluation.
        # Generate the necessary initial data-structures.

        logger.debug("Beginning evaluation")

        quantity_pool = defaultdict(set)  # Dict<Symbol, set<Quantity>>, available Quantity objects.
        plug_in_dict = defaultdict(set)  # Dict<Quantity, set<Model>>, where the Quantities have been plugged in.
        output_dict = defaultdict(set)  # Dict<Quantity, set<Model>>, where the Quantities have been generated.
        candidate_models = set()  # set<Model>, which could generate additional outputs.

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

                logger.debug("Checking model %s", model.title)
                # logger.debug("Quantity pool contains %s quantities:",
                #     len(list(chain.from_iterable(quantity_pool.values()))))

                for property_input_sets in model.evaluation_list:

                    logger.debug("\tGenerating input sets for: %s",
                                 property_input_sets)

                    input_sets = self.generate_input_sets(
                        property_input_sets, quantity_pool)

                    for input_set in input_sets:

                        logger.debug("\t\tEvaluating input set: %s", input_set)

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

                        evaluate_set = {symbol: quantity.value
                                        for symbol, quantity in input_set.items()}
                        output = model.evaluate(evaluate_set)
                        success = output.pop('successful')
                        if not success:
                            logger.debug("Model %s unsuccessful: %s",
                                         model.name, output['message'])
                            continue

                        # input_set led to output from the Model -- gather output
                        #                       -- add output to the graph
                        #                       -- add additional candidate models

                        logger.debug("\t\t\tInput set produced successful output.")
                        continue_loop = True
                        for symbol, quantity in output.items():
                            st = self._symbol_types.get(symbol)
                            if not st:
                                raise ValueError(
                                    "Symbol type {} not found".format(symbol))
                            for m in self._input_to_model[st]:
                                new_models.add(m)
                            q = Quantity(st, quantity)

                            # Set the provenance of the derived quantity
                            q_prov = ProvenanceElement(model=model)
                            q_child = list()
                            for item in input_set.items():
                                q_child.append(item[1])
                            q_prov.inputs = q_child
                            q._provenance = q_prov

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

        if not isinstance(material, CompositeMaterial):
            raise Exception("material provided is not a SuperMaterial: " + str(type(material)))

        # Evaluate material's sub-materials

        evaluated_materials = list()
        for m in material.materials:
            logger.debug("Evaluating sub-material: " + str(id(m)))
            if isinstance(m, CompositeMaterial):
                evaluated_materials.append(self.super_evaluate(m, property_type=property_type))
            else:
                evaluated_materials.append(self.evaluate(m, property_type=property_type))

        # Run all SuperModels in the graph on this SuperMaterial if a material mapping can be established.
        # Store any derived quantities.

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

            if len(mat_mappings) != 1:      # Avoid ambiguous or impossible mappings, at least for now.
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

                    evaluate_set = {symbol: quantity.value
                                    for symbol, quantity in input_set.items()}
                    output = model.evaluate(evaluate_set)
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
                        q = Quantity(st, quantity)
                        to_return._symbol_to_quantity[st].add(q)
                        logger.debug("\t\t\tNew output: " + str(q))

        # Evaluate the SuperMaterial's quantities and return the result.
        mappings = self.evaluate(to_return)._symbol_to_quantity
        to_return._symbol_to_quantity = mappings
        return to_return
