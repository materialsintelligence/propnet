"""
Module containing classes and methods for Model functionality in Propnet code.
"""

import os
import re
import logging
from abc import ABC, abstractmethod
from itertools import chain
from glob import glob
from copy import copy

from monty.serialization import loadfn
from monty.json import MSONable
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from propnet import ureg
from propnet.core.exceptions import ModelEvaluationError
from propnet.symbols import DEFAULT_UNITS

logger = logging.getLogger(__name__)

# TODO: maybe this should go somewhere else, like a dedicated settings.py
TEST_DATA_LOC = os.path.join(os.path.dirname(__file__), "..",
                             "models", "test_data")

# General TODOs:
# TODO: I'm not sure that symbol_map needs to be present in all models,
#       maybe just equation models/relegated to that plug_in method
# TODO: The evaluate/plug_in dichotomy is a big confusing here
#       I suspect they can be consolidated
# TODO: Does the unit_map really need to be specified?  Why can't
#       pint handle this?
class Model(ABC):
    """
    Abstract model class for all models appearing in Propnet

    Args:
        name (str): title of the model
        connections (dict): list of connections dictionaries, which take
            the form {"inputs": [Symbols], "outputs": [Symbols]}, e. g.:
            connections = [{"inputs": ["p", "T"], "outputs": ["V"]},
                           {"inputs": ["T", "V"], "outputs": ["p"]}]
        constraints ([str or Constraint]): string expressions or
            Constraint objects of some condition on which the model is
            valid, e. g. "n > 0", note that this must include symbols if
            there is a symbol_property_map
        description (str): long form description of the model
        categories ([str]): list of categories applicable to
            the model
        references ([str]): list of the informational links
            explaining / supporting the model
        symbol_property_map ({str: str}): mapping of symbols enumerated
            in the plug-in method to canonical symbols, e. g.
            {"n": "index_of_refraction"} etc.
        unit_map ({str: str}): mapping of units to be used in model
            before evaluation of plug_in in evaluate, defaults to
            symbol defaults
    """
    def __init__(self, name, connections, constraints=None,
                 description=None, categories=None, references=None,
                 symbol_property_map=None, unit_map=None):
        self.name = name
        self.connections = connections
        self.description = description
        self.categories = categories
        self.references = references
        # symbol property map initialized as symbol->symbol, then updated
        # with any customization of symbol to properties mapping
        self.symbol_property_map = {}
        self.symbol_property_map = {k: k for k in self.all_properties}
        self.symbol_property_map.update(symbol_property_map or {})

        # Use hard-coded units for properties unless otherwise specified
        if unit_map:
            self.unit_map = unit_map
        else:
            self.unit_map = {prop_name: DEFAULT_UNITS.get(prop_name)
                             for prop_name in self.all_properties}

        # Define constraints by constraint objects or invoke from strings
        constraints = constraints or []
        self.constraints = []
        for constraint in constraints:
            if isinstance(constraint, Constraint):
                self.constraints.append(constraint)
            else:
                self.constraints.append(Constraint(constraint))

    @abstractmethod
    def plug_in(self, symbol_value_dict):
        """
        Plugs in a symbol to quantity dictionary

        Args:
            symbol_value_dict ({symbol: value}): a mapping
                of symbols to values to be substituted
                into the model to yield output

        Returns:
            dictionary of output symbols with associated
                values generated from the input
        """
        return

    def map_properties_to_symbols(self, properties):
        """
        Helper method to convert property-keyed dictionary or list to
        symbol-keyed dictionary or list

        Args:
            properties (list or dict): list of properties or property-
                keyed dictionary

        Returns (list or dict):
            list of symbols or symbol-keyed dict
        """
        rev_map = {v: k for k, v in self.symbol_property_map.items()}
        return remap(properties, rev_map)

    def map_symbols_to_properties(self, symbols):
        """
        Helper method to convert symbol-keyed dictionary or list to
        property-keyed dictionary or list

        Args:
            symbols (list or dict): list of symbols or symbol-keyed
                dictionary

        Returns (list or dict):
            list of properties or property-keyed dict
        """
        return remap(symbols, self.symbol_property_map)

    # TODO: I'm really not crazy about the "successful" key implementation
    #       preventing model failure using try/except is the path to
    #       the dark side
    def evaluate(self, property_value_dict):
        """
        Given a set of property_values, performs error checking to see
        if the corresponding input symbol_values represents a valid
        input set based on the self.connections() method. If so, returns
        a dictionary representing the value of plug_in applied to the
        input_symbols. The dictionary contains a "successful" key
        representing if plug_in was successful.

        The key distinction between evaluate and plug_in is properties
        in properties out vs. symbols in symbols out.  In addition,
        evaluate also handles any requisite unit_mapping

        Args:
            property_value_dict ({property_name: value}): a mapping of
                property names to values to be substituted

        Returns:
            dictionary of output properties with associated values
            generated from the input, along with "successful" if the
            substitution succeeds
        """
        # Remap symbols and units if symbol map isn't none
        symbol_value_dict = self.map_properties_to_symbols(
            property_value_dict)

        # TODO: Is it really necessary to strip these?
        # TODO: maybe this only applies to pymodels or things with objects?
        # strip units from input and keep for reassignment
        old_units = {}
        for symbol, value in symbol_value_dict.items():
            if isinstance(value, ureg.Quantity):
                if self.unit_map.get(symbol):
                    value = value.to(self.unit_map[symbol])
                symbol_value_dict[symbol] = value.magnitude
                old_units[symbol] = value.units

        # check we support this combination of inputs
        # TODO: this shouldn't be necessary
        input_matches = [set(input_set) == set(property_value_dict)
                         for input_set in self.evaluation_list]
        if not any(input_matches):
            return {
                'successful': False,
                'message': "The {} model cannot generate any outputs for "
                           "these inputs: {}".format(
                    self.name, property_value_dict.keys())}
        # TODO: Remove the try-except functionality, high priority
        try:
            # evaluate is allowed to fail
            out = self.plug_in(symbol_value_dict)
            out['successful'] = True
        except Exception as e:
            logger.debug("Model evaluation unsuccessful %s", e)
            return {
                'successful': False,
                'message': str(e)
            }

        # add units to output
        out = self.map_symbols_to_properties(out)
        for key in out:
            if key == 'successful':
                continue
            out[key] = ureg.Quantity(out[key], self.unit_map.get(key))
        return out

    @property
    def title(self):
        """
        Fancy formatted name, removes underscores and capitalizes
        all words of name

        Returns (str):
            formatted title, e. g. "band_gap_refractive_index"
            becomes "Band Gap Refractive Index"
        """
        return self.name.replace('_', ' ').title()

    # Note that these are formulated in terms of properties
    # rather than symbols
    @property
    def input_sets(self):
        """
        Returns (set): set of input property sets
        """
        return [set(self.map_symbols_to_properties(d['inputs']))
                for d in self.connections]

    @property
    def output_sets(self):
        """
        Returns (set): set of output property sets
        """
        return [set(self.map_symbols_to_properties(d['outputs']))
                for d in self.connections]

    @property
    def all_inputs(self):
        """
        Returns (set): set of all input properties
        """
        return set(chain.from_iterable(self.input_sets))

    @property
    def all_outputs(self):
        """
        Returns (set): set of all output properties
        """
        return set(chain.from_iterable(self.output_sets))

    @property
    def all_properties(self):
        """
        Returns (set): set of all output properties
        """
        return self.all_inputs.union(self.all_outputs)

    # TODO: I think this should be merged with input_sets maybe called
    # relevant properties or something, also shouldn't need both syms
    # and props
    @property
    def evaluation_list(self):
        """
        Gets everything one needs to call the evaluate method, which
        is all of the input properties and constraints

        Returns:
            list of sets of inputs with constraint properties included
        """
        return [set(input_set | self.constraint_properties)
                for input_set in self.input_sets]

    def test(self, inputs, outputs):
        """
        Runs a test of the model to determine whether its operation
        is consistent with the specified inputs and outputs

        Args:
            inputs (dict): set of input names to values
            outputs (dict): set of output names to values

        Returns (bool): True if test succeeds
        """
        model_outputs = self.plug_in(inputs)
        errmsg = "Model does not match known output for {}".format(
            self.name)
        for k, known_output in outputs.items():
            model_output = model_outputs[k]
            # TODO: address as part of unit refactor
            if hasattr(model_output, 'magnitude'):
                model_output = model_output.magnitude
            if isinstance(known_output, (float, list)):
                if not np.allclose(model_output, known_output):
                    raise ModelEvaluationError(errmsg)
            elif model_output != known_output:
                raise ModelEvaluationError(errmsg)
        return True

    def validate_from_preset_test(self):
        """
        Validates from test data based on the model name

        Returns:
            True if validation completes successfully
        """
        test_datasets = self.load_test_data(self.name)
        for test_dataset in test_datasets:
            self.test(**test_dataset)
        return True

    @property
    def constraint_properties(self):
        """
        Returns (set): set of constraint input properties
        """
        # Constraints are defined only in terms of symbols
        all_syms = [self.map_symbols_to_properties(c.all_inputs)
                    for c in self.constraints]
        return set(chain.from_iterable(all_syms))

    def check_constraints(self, input_properties):
        """
        Checks the constraints based on input property set

        Args:
            input_properties ({property: value}): property value
                dictionary for input to constraints

        Returns (bool):
            True if constraints are satisfied, false if not
        """
        input_symbols = self.map_properties_to_symbols(input_properties)
        for constraint in self.constraints:
            if not constraint.plug_in(input_symbols):
                return False
        return True

    @staticmethod
    def load_test_data(name, test_data_loc=TEST_DATA_LOC):
        """
        Loads test data from preset or specified directory.
        Finds a json or yaml file with the prefix "name" and
        loads it.

        Args:
            name (str): name for test data to load
            test_data_loc (str): directory location for test data

        Returns (dict):
            Dictionary of test data
        """
        filelist = glob(os.path.join(test_data_loc, "{}.*".format(name)))
        # Raise error if 0 files or more than 1 file
        if len(filelist) != 1:
            raise ValueError("{} test data files for {}".format(
                len(filelist), name))
        return loadfn(filelist[0])

    @property
    def example_code(self):
        """
        Generates example code from test data, useful for
        documentation.

        Returns: example code for this model

        """
        test_data = self.load_test_data(self.name)


        example_inputs = test_data[0]['inputs']
        example_outputs = str(test_data[0]['outputs'])

        symbol_definitions = []
        evaluate_args = []
        for input_name, input_value in example_inputs.items():

            symbol_str = "{input_name} = {input_value}".format(
                input_name=input_name,
                input_value=input_value,
            )
            symbol_definitions.append(symbol_str)

            evaluate_str = "\t'{}': {}".format(input_name, input_name)
            evaluate_args.append(evaluate_str)

        symbol_definitions = '\n'.join(symbol_definitions)
        evaluate_args = '\n'.join(evaluate_args)

        example_code = CODE_EXAMPLE_TEMPLATE.format(
            model_name=self.name,
            symbol_definitions=symbol_definitions,
            evaluate_args=evaluate_args,
            example_outputs=example_outputs)

        return example_code


CODE_EXAMPLE_TEMPLATE = """
from propnet.models import load_default_model

{symbol_definitions}

model = load_default_model("{model_name}")
model.evaluate({{
{evaluate_args}
}})  # returns {example_outputs}
"""


class EquationModel(Model, MSONable):
    """
    Equation model is a Model subclass which is invoked
    from a list of equations

    Args:
        name (str): title of the model
        equations ([str]): list of string equations to parse
        connections ([dict]): list of connections dictionaries,
            which take the form {"inputs": [Symbols], "outputs": [Symbols]},
            for example:
            connections = [{"inputs": ["p", "T"], "outputs": ["V"]},
                           {"inputs": ["T", "V"], "outputs": ["p"]}]
        constraints ([str]): constraints on models
        description (str): long form description of the model
        categories (str): list of categories applicable to
            the model
        references ([str]): list of the informational links
            explaining / supporting the model

    """
    def __init__(self, name, equations, connections, constraints=None,
                 symbol_property_map=None, description=None,
                 categories=None, references=None, unit_map=None):
        self.equations = equations
        super(EquationModel, self).__init__(
            name, connections, constraints, description,
            categories, references, symbol_property_map, unit_map)

    # TODO: shouldn't this respect/use connections info,
    #       or is that done elsewhere?
    def plug_in(self, symbol_value_dict):
        """
        Equation plug-in solves the equation for all input
        and output combinations, returning the corresponding
        output values

        Args:
            symbol_value_dict ({symbol: value}): symbol-keyed
                dict of values to be substituted

        Returns (dict):
            symbol-keyed output dictionary
        """
        # Parse equations and substitute
        eqns = [parse_expr(eq) for eq in self.equations]
        eqns = [eqn.subs(symbol_value_dict) for eqn in eqns]
        possible_outputs = set()
        for eqn in eqns:
            possible_outputs = possible_outputs.union(eqn.free_symbols)
        outputs = {}
        # Determine outputs from solutions to substituted equations
        for possible_output in possible_outputs:
            solutions = sp.nonlinsolve(eqns, possible_output)
            # taking first solution only, and only asking for one output symbol
            # so know length of output tuple for solutions will be 1
            solution = list(solutions)[0][0]
            if not isinstance(solution, sp.EmptySet):
                outputs[str(possible_output)] = float(sp.N(solution))
        return outputs

    @classmethod
    def from_file(cls, filename):
        """
        Invokes EquationModel from filename

        Args:
            filename (str): filename containing model

        Returns:
            Model corresponding to contents of file
        """
        model = loadfn(filename)
        if isinstance(model, dict):
            return cls.from_dict(model)
        return model


class PyModel(Model):
    """
    Purely python based model which allows for a flexible "plug_in"
    method as input, then invokes that method in the defined plug-in
    method
    """
    def __init__(self, name, connections, plug_in, constraints=None,
                 description=None, categories=None, references=None,
                 symbol_property_map=None, unit_map=None):
        self._plug_in = plug_in
        super(PyModel, self).__init__(
            name, connections, constraints, description,
            categories, references, symbol_property_map, unit_map)

    def plug_in(self, symbol_value_dict):
        """
        plug_in for PyModel uses the attached _plug_in attribute
        as a method with the input symbol_value_dict

        Args:
            symbol_value_dict ({symbol: value}): dict containing
                symbol-keyed values to substitute

        Returns:
            value of substituted expression
        """
        return self._plug_in(symbol_value_dict)


# Note that this class exists purely as a factory method for PyModel
# which could be implemented as a class method of PyModel
# but wouldn't serialize as cleanly
class PyModuleModel(PyModel):
    """
    PyModuleModel is a class instantiated by a model path only,
    which exists primarily for the purpose of serializing python models
    """
    def __init__(self, module_path):
        """
        Args:
            module_path (str): path to module to instantiate model
        """
        self._module_path = module_path
        mod = __import__(module_path, globals(), locals(), ['config'], 0)
        super(PyModuleModel, self).__init__(**mod.config)

    def as_dict(self):
        return {"module_path": self._module_path,
                "@module": "propnet.core.model",
                "@class": "PyModuleModel"}


# TODO: filter might be unified with constraint
# TODO: the implementation here is inherently difficult because
#       it relies on iterative pairing.  A lookup-oriented strategy
#       might be implemented in the future.
class CompositeModel(Model):
    def __init__(self, name, connections, pre_filter=None,
                 filter=None, **kwargs):
        """
        Args:
            name: title of the model
            connections (dict): list of connections dictionaries,
                which take the form {"inputs": [Symbols],
                                     "outputs": [Symbols]}, e. g.:
                connections = [{"inputs": ["p", "T"], "outputs": ["V"]},
                               {"inputs": ["T", "V"], "outputs": ["p"]}]
            pre_filter (callable): callable for filtering the input materials
                into independent sets which are supplied as candidates
                for inputs into the model, Defaults to None.  Signature
                should be pre_filter(materials) where materials is a list
                of materials, and returns a dictionary with the materials
                keyed by the associated arguments to plug_in e. g.
                {'metal': [Materials], 'oxide': [Materials]}
            filter (callable): callable for filtering pairs of materials
                to ensure that valid pairings are supplied.  Takes
                an input dictionary of key-value pairs corresponding to
                input material candidates keyed by input kwarg to plug_in,
                e. g. filter({'metal': Material, 'oxide': Material})
            **kwargs: model params, e. g. description, references, etc.
        """
        mat_inputs = {arg.split('.')[0] for arg in connections['inputs']
                      if '.' in arg}
        self.n_materials = len(mat_inputs)
        self.pre_filter = pre_filter
        self.filter = filter
        super(CompositeModel, self).__init__(name=name, connections=connections, **kwargs)


class PyModuleCompositeModel(CompositeModel):
    """
    PyModuleModel is a class instantiated by a model path only,
    which exists primarily for the purpose of serializing python models
    """
    def __init__(self, module_path):
        """
        Args:
            module_path (str): path to module to instantiate model
        """
        self._module_path = module_path
        mod = __import__(module_path, globals(), locals(), ['config'], 0)
        super(PyModuleCompositeModel, self).__init__(**mod.config)

    def as_dict(self):
        return {"module_path": self._module_path,
                "@module": "propnet.core.model",
                "@class": "PyModuleCompositeModel"}


# Right now I don't see much of a use case for pythonic functionality
# here but maybe there should be
# TODO: this could use a bit more finesse
class ConstraintInterface:
    def plug_in(self, symbol_value_dict):
        """
        ABSTRACT
        Method analogous to that of the Model class.
        In this case the method contract demands a boolean
        be returned representing whether the constraint was
        met.
        Args:
            symbol_value_dict ({symbol: value}): dict containing
                symbol-keyed values to substitute
        Returns:
            (bool) true/false representing whether the constraint
                   was met.
        """
        pass

class Constraint(Model, ConstraintInterface):
    """
    Constraint class, resembles a model, but should outputs
    true or false based on a string expression containing
    input symbols
    """
    def __init__(self, expression, name=None, **kwargs):
        """
        Args:
            expression (str): str to be parsed to evaluate constraint
            name (str): optional name for constraint, default None
            **kwargs: kwargs for model
        """
        self.expression = expression.replace(' ', '')
        # Parse all the non-math symbols and assign to inputs
        split = re.split("[+-/*<>=()]", self.expression)
        inputs = [s for s in split if not will_it_float(s) and s]
        connections = [{"inputs": inputs, "outputs": ["is_valid"]}]
        Model.__init__(
            self, name=name, connections=connections, **kwargs)

    def plug_in(self, symbol_value_dict):
        """
        Evaluates the expression with sympy and provided values
        and returns the boolean of that expression

        Args:
            symbol_value_dict ({symbol: value}): dict containing
                symbol-keyed values to substitute

        Returns:
            value of substituted expression
        """
        return parse_expr(self.expression, symbol_value_dict)

    def __repr__(self):
        return "Constraint: {}".format(self.expression)


def will_it_float(input_to_test):
    """
    Helper function to determine if input string can be cast to float

    Args:
        input_to_test (str): input string to be tested
    """
    try:
        float(input_to_test)
        return True
    except ValueError:
        return False


def remap(dict_or_list, mapping):
    """
    Helper method to remap entries in a list or keys in a dictionary
    based on an input map, used to translate symbols to properties
    and vice-versa

    Args:
        dict_or_list ([] or {}) a list of properties or property-keyed
            dictionary to be remapped using symbols.
        mapping ({}): dictionary of values to remap

    Returns:
        remapped list of items or item-keyed dictionary
    """
    output = copy(dict_or_list)
    if isinstance(output, dict):
        for in_key, out_key in mapping.items():
            if in_key in output:
                output[out_key] = output.pop(in_key)
    else:
        for idx, in_item in enumerate(output):
            out_item = mapping.get(in_item)
            if out_item:
                output[idx] = out_item
    return output
