"""
Module containing classes and methods for Model functionality in Propnet code.
"""

import os
import re
import logging
from abc import ABC, abstractmethod
from itertools import chain

import six
from monty.serialization import loadfn
from monty.json import MSONable, MontyDecoder
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr


from propnet.core.exceptions import ModelEvaluationError, SymbolConstraintError
from propnet.core.quantity import QuantityFactory, NumQuantity
from propnet.core.utils import references_to_bib, PrintToLogger
from propnet.core.provenance import ProvenanceElement
from propnet.symbols import DEFAULT_UNITS

logger = logging.getLogger(__name__)

# TODO: maybe this should go somewhere else, like a dedicated settings.py
TEST_DATA_LOC = os.path.join(os.path.dirname(__file__), "..",
                             "models", "test_data")


class Model(ABC):
    """
    Abstract model class for all models appearing in Propnet

    Args:
        name (str): title of the model
        connections ([dict]): list of connections dictionaries, which take
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
        implemented_by ([str]): list of authors of the model by their
            github usernames
        symbol_property_map ({str: str}): mapping of symbols enumerated
            in the plug-in method to canonical symbols, e. g.
            {"n": "index_of_refraction"} etc.
        scrub_units (bool or {str: str}): whether or not units should
            be scrubbed in evaluation procedure, if a boolean is specified,
            quantities are converted to default units before scrubbing, if
            a dict, quantities are specified to units corresponding to the
            unit assigned to the symbol in the dicts.  Units are scrubbed
            by default for PyModels/PyModule models and EquationModels with
            'empirical' categories
        test_data (list of {'inputs': [], 'outputs': []): test data with
            which to evaluate the model
    """
    def __init__(self, name, connections, constraints=None,
                 description=None, categories=None, references=None, implemented_by=None,
                 symbol_property_map=None, scrub_units=None, test_data=None):
        self.name = name
        self.connections = connections
        self.description = description
        self.categories = categories or []
        self.implemented_by = implemented_by or []
        self.references = references_to_bib(references or [])
        # symbol property map initialized as symbol->symbol, then updated
        # with any customization of symbol to properties mapping
        self.symbol_property_map = {k: k for k in self.all_properties}
        self.symbol_property_map.update(symbol_property_map or {})

        if scrub_units is not None or 'empirical' in self.categories:
            self.unit_map = {prop_name: DEFAULT_UNITS.get(prop_name)
                             for prop_name in self.all_properties}
            # Update with explicitly supplied units if specified
            if isinstance(scrub_units, dict):
                self.unit_map.update(self.map_symbols_to_properties(scrub_units))
            self.unit_map = self.map_properties_to_symbols(self.unit_map)
        else:
            self.unit_map = {}

        # Define constraints by constraint objects or invoke from strings
        constraints = constraints or []
        self.constraints = []
        for constraint in constraints:
            if isinstance(constraint, Constraint):
                self.constraints.append(constraint)
            else:
                self.constraints.append(Constraint(constraint))

        self._test_data = test_data or self.load_test_data()

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
        rev_map = {v: k for k, v in getattr(self, "symbol_property_map", {}).items()}
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
        return remap(symbols, getattr(self, "symbol_property_map", {}))

    def evaluate(self, symbol_quantity_dict, allow_failure=True):
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
            symbol_quantity_dict ({property_name: Quantity}): a mapping of
                symbol names to quantities to be substituted
            allow_failure (bool): whether or not to catch
                errors in model evaluation

        Returns:
            dictionary of output properties with associated values
            generated from the input, along with "successful" if the
            substitution succeeds
        """
        # Remap symbols and units if symbol map isn't none
        symbol_quantity_dict = self.map_properties_to_symbols(
            symbol_quantity_dict)

        for (k, v) in symbol_quantity_dict.items():
            replacing = self.symbol_property_map.get(k, k)
            # to_quantity() returns original object if it's already a BaseQuantity
            # unlike Quantity() which will return a deep copy
            symbol_quantity_dict[k] = QuantityFactory.to_quantity(replacing, v)

        # TODO: Is it really necessary to strip these?
        # TODO: maybe this only applies to pymodels or things with objects?
        # strip units from input and keep for reassignment
        symbol_value_dict = {}

        for symbol, quantity in symbol_quantity_dict.items():
            # If unit map convert and then scrub units
            if self.unit_map.get(symbol):
                quantity = quantity.to(self.unit_map[symbol])
                symbol_value_dict[symbol] = quantity.magnitude
            # Otherwise use values
            else:
                symbol_value_dict[symbol] = quantity.value

        contains_complex_input = any(NumQuantity.is_complex_type(v) for v in symbol_value_dict.values())
        # Plug in and check constraints
        try:
            with PrintToLogger():
                out = self.plug_in(symbol_value_dict)
        except Exception as err:
            if allow_failure:
                return {"successful": False,
                        "message": "{} evaluation failed: {}".format(self, err)}
            else:
                raise err
        if not self.check_constraints({**symbol_value_dict, **out}):
            return {"successful": False,
                    "message": "Constraints not satisfied"}

        provenance = ProvenanceElement(
            model=self.name, inputs=list(symbol_quantity_dict.values()),
            source="propnet")

        out = self.map_symbols_to_properties(out)
        for symbol, value in out.items():
            try:
                quantity = QuantityFactory.create_quantity(symbol, value, self.unit_map.get(symbol),
                                                           provenance=provenance)

            except SymbolConstraintError as err:
                if allow_failure:
                    errmsg = "{} symbol constraint failed: {}".format(self, err)
                    return {"successful": False,
                            "message": errmsg}
                else:
                    raise err

            if quantity.contains_nan_value():
                return {"successful": False,
                        "message": "Evaluation returned invalid values (NaN)"}
            # TODO: Update when we figure out how we're going to handle complex quantities
            # Model evaluation will fail if complex values are returned when no complex input was given
            # Can surely handle this more gracefully, or assume that the users will apply constraints
            if quantity.contains_imaginary_value() and not contains_complex_input:
                return {"successful": False,
                        "message": "Evaluation returned invalid values (complex)"}

            out[symbol] = quantity

        out['successful'] = True
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

    # Note: these are formulated in terms of properties rather than symbols
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
        Returns (set): set of all properties
        """
        return self.all_inputs.union(self.all_outputs)

    @property
    def evaluation_list(self):
        """
        Gets everything one needs to call the evaluate method, which
        is all of the input properties and constraints

        Returns:
            list of sets of inputs with constraint properties included
        """
        return [list(inputs | self.constraint_properties - outputs)
                for inputs, outputs in zip(self.input_sets, self.output_sets)]

    def test(self, inputs, outputs):
        """
        Runs a test of the model to determine whether its operation
        is consistent with the specified inputs and outputs

        Args:
            inputs (dict): set of input names to values
            outputs (dict): set of output names to values

        Returns (bool): True if test succeeds
        """
        evaluate_inputs = self.map_symbols_to_properties(inputs)
        evaluate_inputs = {s: QuantityFactory.create_quantity(s, v, self.unit_map.get(s))
                           for s, v in evaluate_inputs.items()}
        evaluate_outputs = self.evaluate(evaluate_inputs, allow_failure=False)
        evaluate_outputs = self.map_properties_to_symbols(evaluate_outputs)
        errmsg = "{} model test failed on ".format(self.name) + "{}\n"
        errmsg += "{}(test data) = {}\n"#.format(k, known_output)
        errmsg += "{}(model output) = {}"#.format(k, plug_in_output)
        for k, known_output in outputs.items():
            symbol = self.symbol_property_map[k]
            units = self.unit_map.get(k)
            known_quantity = QuantityFactory.create_quantity(symbol, known_output, units)
            evaluate_output = evaluate_outputs[k]
            if isinstance(known_quantity, NumQuantity) or isinstance(known_quantity.value, list):
                if not np.allclose(known_quantity.value, evaluate_output.value):
                    errmsg = errmsg.format("evaluate", k, evaluate_output,
                                           k, known_quantity)
                    raise ModelEvaluationError(errmsg)
            elif known_quantity != evaluate_output:
                errmsg = errmsg.format("evaluate", k, evaluate_output,
                                       k, known_quantity)
                raise ModelEvaluationError(errmsg)

        return True

    def validate_from_preset_test(self):
        """
        Validates from test data based on the model name

        Returns:
            True if validation completes successfully
        """
        for test_dataset in self._test_data:
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

    def load_test_data(self, test_data_path=None, deserialize=True):
        """
        Loads test data from preset or specified directory.
        Finds a json or yaml file with the prefix "name" and
        loads it.

        Args:
            test_data_path (str): test data file location
            deserialize (bool): whether or not to deserialize the test
                data, primarily used for printing example code

        Returns (dict):
            Dictionary of test data
        """
        if test_data_path is None:
            test_data_path = os.path.join(TEST_DATA_LOC,
                                          "{}.json".format(self.name))
        if os.path.exists(test_data_path):
            cls = MontyDecoder if deserialize else None
            return loadfn(test_data_path, cls=cls)

    @property
    def example_code(self):
        """
        Generates example code from test data, useful for
        documentation.

        Returns: example code for this model

        """
        example_inputs = self._test_data[0]['inputs']
        example_outputs = str(self._test_data[0]['outputs'])

        symbol_definitions = []
        evaluate_args = []
        imports = []
        for input_name, input_value in example_inputs.items():

            if hasattr(input_value, 'as_dict'):
                input_value = input_value.as_dict()
                # temp fix for ComputedEntry pending pymatgen fix
                if 'composition' in input_value:
                    input_value['composition'] = dict(input_value['composition'])

            if isinstance(input_value, dict) and input_value.get("@module"):
                input_value_string = "{}.from_dict({})".format(
                    input_value['@class'], input_value)
                imports += ["from {} import {}".format(
                    input_value['@module'], input_value['@class'])]
            elif isinstance(input_value, six.string_types):
                input_value_string = '"{}"'.format(input_value)
            elif isinstance(input_value, np.ndarray):
                input_value_string = input_value.tolist()
            else:
                input_value_string = input_value
            symbol_str = "{input_name} = {input_value}".format(
                input_name=input_name,
                input_value=input_value_string,
            )
            symbol_definitions.append(symbol_str)

            evaluate_str = "\t'{}': {},".format(input_name, input_name)
            evaluate_args.append(evaluate_str)

        symbol_definitions = '\n'.join(symbol_definitions)
        evaluate_args = '\n'.join(evaluate_args)

        example_code = CODE_EXAMPLE_TEMPLATE.format(
            model_name=self.name,
            imports='\n'.join(imports),
            symbol_definitions=symbol_definitions,
            evaluate_args=evaluate_args,
            example_outputs=example_outputs)

        return example_code

    def __str__(self):
        return "Model: {}".format(self.name)

    def __repr__(self):
        return self.__str__()


CODE_EXAMPLE_TEMPLATE = """
from propnet.models import {model_name}
{imports}

{symbol_definitions}

{model_name}.plug_in({{
{evaluate_args}
}})
\"\"\"
returns {example_outputs}
\"\"\"
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
            If no connections are specified (as default), EquationModel
            attempts to find them by the convention that the symbol
            in front of the equals sign is the output and the symbols
            to the right are the inputs, e. g. OUTPUT = INPUT_1 + INPUT_2,
            alternatively, using solve_for_all_symbols will derive all
            possible input-output connections
        constraints ([str]): constraints on models
        description (str): long form description of the model
        categories (str): list of categories applicable to
            the model
        references ([str]): list of the informational links
            explaining / supporting the model
        test_data (list of {'inputs': [], 'outputs': []): test data with
            which to evaluate the model

    """
    def __init__(self, name, equations, connections=None, constraints=None,
                 symbol_property_map=None, description=None,
                 categories=None, references=None, implemented_by=None,
                 scrub_units=None, solve_for_all_symbols=False, test_data=None):

        self.equations = equations
        sympy_expressions = [parse_expr(eq.replace('=', '-(')+')')
                             for eq in equations]
        # If no connections specified, derive connections
        if connections is None:
            connections = []
            if solve_for_all_symbols:
                connections, equations = [], []
                for expr in sympy_expressions:
                    for symbol in expr.free_symbols:
                        new = sp.solve(expr, symbol)
                        inputs = get_syms_from_expression(new)
                        connections.append(
                            {"inputs": inputs,
                             "outputs": [str(symbol)],
                             "_lambdas": {str(symbol): sp.lambdify(inputs, new)}
                             })
            else:
                for eqn in equations:
                    output_expr, input_expr = eqn.split('=')
                    inputs = get_syms_from_expression(input_expr)
                    outputs = get_syms_from_expression(output_expr)
                    connections.append(
                        {"inputs": inputs,
                         "outputs": outputs,
                         "_lambdas": {outputs[0]: sp.lambdify(inputs, parse_expr(input_expr))}
                         })
        else:
            # TODO: I don't think this needs to be supported necessarily
            #       but it's causing problems with models with one input
            #       and two outputs where you only want one connection
            for connection in connections:
                new = sp.solve(sympy_expressions, connection['outputs'])
                lambdas = {str(sym): sp.lambdify(connection['inputs'], solved)
                           for sym, solved in new.items()}
                connection["_lambdas"] = lambdas

        #self.equations = equations
        super(EquationModel, self).__init__(
            name, connections, constraints, description,
            categories, references, implemented_by,
            symbol_property_map, scrub_units,
            test_data=test_data)

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
        output = {}
        for connection in self.connections:
            if set(connection['inputs']) <= set(symbol_value_dict.keys()):
                for output_sym, func in connection['_lambdas'].items():
                    output_vals = func(**symbol_value_dict)
                    # TODO: this decision to only take max real values should
                    #       should probably be reevaluated at some point
                    # Scrub nan values and take max
                    if isinstance(output_vals, list):
                        output_val = max([v for v in output_vals
                                          if not isinstance(v, complex)])
                    else:
                        output_val = output_vals
                    output.update({output_sym: output_val})
        if not output:
            raise ValueError("No valid input set found in connections")
        else:
            return output

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
    method.  Note that PyModels scrub units by default, in contrast
    to EquationModels
    """
    def __init__(self, name, connections, plug_in, constraints=None,
                 description=None, categories=None, references=None,
                 implemented_by=None, symbol_property_map=None,
                 scrub_units=True, test_data=None):
        self._plug_in = plug_in
        super(PyModel, self).__init__(
            name, connections, constraints, description,
            categories, references, implemented_by,
            symbol_property_map, scrub_units,
            test_data=test_data)

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
class CompositeModel(PyModel):
    """
    Model based on deriving emerging properties from collections of
    materials of known properties.

    Model requires the unambiguous assignment of materials to labels.
    These labeled materials' properties are then referenced.
    """

    def __init__(self, name, connections, plug_in, pre_filter=None,
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
        PyModel.__init__(self, name=name, connections=connections,
                         plug_in=plug_in, **kwargs)
        self.pre_filter = pre_filter
        self.filter = filter
        self.mat_inputs = []
        for connection in connections:
            for input in connection['inputs']:
                mat = CompositeModel.get_material(input)
                if mat is not None:
                    self.mat_inputs.append(mat)

    @staticmethod
    def get_material(input):
        """
        Args:
            input (String): inputs entry from the connections instance variable.
        Returns:
            String or None only material identifiers from the input argument.
        """
        separation = input.split('.')
        components = len(separation)
        if components == 2:
            return separation[0]
        elif components > 2:
            raise Exception('Connections can only contain 1 period separator.')
        return None

    @staticmethod
    def get_symbol(input):
        """
        Args:
            input (String): inputs entry from the connections instance variable.
        Returns:
            String only symbol identifiers from the input argument.
        """
        separation = input.split('.')
        components = len(separation)
        if components == 1:
            return input
        elif components == 2:
            return separation[1]
        elif components > 2:
            raise Exception('Connections can only contain 1 period separator.')
        return None

    def gen_material_mappings(self, materials):
        """
        Given a set of materials, returns a mapping from each material label
        found in self.connections to a Material object.
        Args:
            materials (list<Material>): list of candidate Material objects.
        Returns:
            (list<dict<String, Material>>) mapping from material label
                to Material object.
        """
        # Group by label all possible candidate materials in a
        # dict<String, list<Material>>
        pre_process = dict()
        for material in self.mat_inputs:
            pre_process[material] = None
        if self.pre_filter:
            cache = self.pre_filter(materials)
            for key in cache.keys():
                if key not in pre_process.keys():
                    raise Exception("pre_filter method returned unrecognized "
                                    "material name.")
                val = cache[key]
                if not isinstance(val, list):
                    raise Exception("pre_filter method did not return a list "
                                    "of candidate materials.")
                pre_process[key] = val
        for key in pre_process.keys():
            if pre_process[key] is None:
                pre_process[key] = materials

        # Check if any combinations are possible - return if not.
        for material in pre_process.keys():
            if len(pre_process[material]) == 0:
                return []

        # Create all combinatorial pairs.
        ## Setup helper variables.
        to_return = []
        tracking = [0]*len(pre_process.keys())
        materials = []
        for material in pre_process.keys():
            materials.append(material)
        overflow = len(tracking)

        ## Setup helper functions.
        def gen_mat_mapping():
            """Generates a material mapping given current state"""
            to_return = dict()
            for i in range(0, len(materials)):
                to_return[materials[i]] = pre_process.get(materials[i])[tracking[i]]
            return to_return

        def step():
            """Advances the state, returns a boolean as to whether the loop should continue"""
            i = 0
            while i < len(tracking) and inc(i):
                i += 1
            return i != len(tracking)

        def inc(index):
            """Increments an index in tracking. Returns if index had to wrap back to zero."""
            tracking[index] += 1
            tracking[index] %= len(pre_process[materials[index]])
            return tracking[index] == 0

        continue_loop = True
        while continue_loop:
            # Generate the material mapping.
            cache = gen_mat_mapping()
            # Queue up next iteration.
            continue_loop = step()
            # Check for duplicate materials in the set - don't include these
            duplicates = False
            vals = [v for v in cache.values()]
            for i in range(0, len(vals)):
                if not duplicates:
                    for j in range(i+1, len(vals)):
                        if vals[i] is vals[j]:
                            duplicates = True
                            break
            if duplicates:
                continue
            # Check if materials set is valid
            if not self.filter(cache):
                continue
            # Accept the input set.
            to_return.append(cache)

        return to_return


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
class Constraint(Model):
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

    "If she weights the same as a duck... she's made of wood"

    Args:
        input_to_test (str): input string to be tested
    """
    try:
        float(input_to_test)
        return True
    except ValueError:
        return False


def remap(obj, mapping):
    """
    Helper method to remap entries in a list or keys in a dictionary
    based on an input map, used to translate symbols to properties
    and vice-versa

    Args:
        obj ([] or {} or set) a list of properties or property-keyed
            dictionary to be remapped using symbols.
        mapping ({}): dictionary of values to remap

    Returns:
        remapped list of items or item-keyed dictionary
    """
    if isinstance(obj, dict):
        new = {mapping.get(in_key) or in_key: obj.get(in_key)
               for in_key in obj.keys()}
    else:
        new = [mapping.get(in_item) or in_item for in_item in obj]
        if isinstance(obj, set):
            new = set(new)
    return new


def get_syms_from_expression(expression):
    """
    Helper function to get all sympy symbols from a string expression

    Args:
        expression (str or sympy expression): string or sympy expression
    """
    if isinstance(expression, six.string_types):
        expression = parse_expr(expression)
    if isinstance(expression, list):
        out = list(chain.from_iterable([get_syms_from_expression(expr)
                                        for expr in expression]))
    else:
        out = [str(v) for v in expression.free_symbols]
    return list(set(out))
