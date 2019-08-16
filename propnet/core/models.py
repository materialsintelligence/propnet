"""Models representing connections between materials properties.

This module contains classes that represent models, or the way that materials properties
are connected to one another. propnet features several kinds of models:

- *EquationModel*: used for relationships between properties of a single material easily expressed as a
  simple mathematical equation
- *PyModel*: custom Python modules used for more complex relationships between properties of a single material
  not easily expressed with a simple mathematical equation
- *CompositeModel*: Python-based models for calculating properties for mixed/composite materials

These three classes are the main model types used. `PyModuleModel` and `PyModuleCompositeModel` are not intended
for direct instantiation by the user, but are helper classes for constructing `PyModel` objects from the templated
Python modules in propnet's core model library.

The `Constraint` class is meant to function similar to an `EquationModel` and constrains the values of properties
used in a model. They are initialized with an equality or inequality statement that must be satisfied for the input
or output of the model to be considered valid. These `Constraint` objects can be passed to a model in the
``'constraints'`` keyword.

The recommended approach to creating models is by following the template approach, as described in the demo
iPython notebook of the repository (``<root>/demo/Getting Started.ipynb``). Once a YAML (for equation-based models)
or a Python module template (for more complex models or composite material models) is completed, place them in the
correct directory under ``<root>/models/`` and they will be imported with ``import propnet.models``.
However, each model type can be constructed manually. See the individual classes for examples.

Examples:
    There are two methods to run a model on some input data: ``plug_in()`` and ``evaluate()``. The distinction
    between the two methods is in the format of the inputs and how that data is handled. Both methods take dictionaries
    as input arguments.

    As an example, say we have an equation-based model ``model`` which relates some property `symbol_a` to some
    property `symbol_b` by ``B = 2 * A**3`` where `A` is the variable representing the value of `symbol_a` in units
    `unit_of_a` and `B` represents the value of `symbol_b` in `unit_of_b` where ``unit_of_b = unit_of_a**3``.

    For ``plug_in()``, the method expects the input dict's keys to be the variable names in the
    equation or the expected variable names in the Python module. The values are expected to be the raw data, ready
    to be used by the Python module or plugged into an equation without change. So, they must be of the correct Python
    type and unit/dimensionality (for numerical inputs). For our example model:

    >>> model.plug_in({'A': 5})
    {'B': 250}

    The output will be variable keyed with raw values as well. Raw in, raw out.

    For ``evaluate()``, the method expects the input dict's keys to be the symbol names the model requires and the
    values are expected to be Quantity objects of those symbol types. ``evaluate()`` or the Quantity class will take
    care of any data type or unit conversions necessary to run the model. For our example model:

    >>> from propnet.core.quantity import QuantityFactory as QF
    >>> output = model.evaluate({'symbol_a': QF.create_quantity('symbol_a', 5, 'unit_of_a')})
    >>> print(output)
    {'symbol_b': <symbol_b, 250 unit_of_b, []>, 'successful': True}
    >>> type(output['symbol_b'])
    propnet.core.quantity.NumQuantity

    The output will be symbol indexed with Quantity outputs and a flag to indicate if the model was successfully
    evaluated. ``evaluate()`` also has more sophisticated checks to ensure that the input and output is valid and
    does not violate any constraints placed on the model or the symbols that it is using.

    For example, if `A` could have different units `other_a_unit` where ``unit_of_a = 100 * other_a_unit``,
    ``evaluate()`` would know how to handle the unit conversion, if needed, whereas ``plug_in()`` does not.

    >>> model.evaluate({'symbol_a': QF.create_quantity('symbol_a', 0.05, 'other_a_unit')})
    {'symbol_b': <symbol_b, 250 unit_of_b, []>, 'successful': True}
    >>> model.plug_in({'A': 0.05})
    {'B': 0.00025}

    If the model requires the inputs be in certain units to be evaluated correctly (i.e. if your model is empirical),
    models can be designed to convert units automatically upon calling ``evaluate()`` by invoking the
    ``units_for_evaluation`` keyword upon instantiation.
"""

import os
import re
import logging
from abc import ABC, abstractmethod
from itertools import chain
import warnings
import six
from copy import deepcopy

from monty.serialization import loadfn
from monty.json import MSONable, MontyDecoder
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from pint import DimensionalityError, UnitStrippedWarning

from propnet.core.exceptions import ModelEvaluationError, SymbolConstraintError
from propnet.core.quantity import QuantityFactory, NumQuantity, BaseQuantity
from propnet.core.utils import references_to_bib, PrintToLogger
from propnet.core.provenance import ProvenanceElement
from propnet import ureg
from propnet.core.registry import Registry

logger = logging.getLogger(__name__)

# TODO: maybe this should go somewhere else, like a dedicated settings.py
TEST_DATA_LOC = os.path.join(os.path.dirname(__file__), "..",
                             "models", "tests", "pymodel_test_data")


class Model(ABC):
    """
    Abstract model class for all models appearing in propnet. All models will have the attributes and
    functions described here, although some will behave differently based on implementation.

    Attributes:
        name (str): unique name for the model
        description (str): information about the model and its origin, constraints, etc. This information
            is displayed on the website.
        display_names (`list` of `str`): nicely formatted name(s) of the model
        categories (`list` of `str`): metadata categorizing the model ("empirical", "electronic",
            "mechanical", etc.)
        implemented_by (`list` of `str`): list of authors (GitHub usernames, preferably) who implemented
            the model
        references (`list` of `str`): BibTeX strings of scholarly references for the model
        constraints (`list` of `propnet.core.models.Constraint`): list of Constraint objects representing
            requirements that must be met for inputs/outputs to the model
    """

    _registry_name = "models"
    """str: name of the Registry in which to register instances of this class"""

    def __init__(self, name, connections, constraints=None, display_names=None,
                 description=None, categories=None, references=None, implemented_by=None,
                 variable_symbol_map=None, units_for_evaluation=None, test_data=None,
                 is_builtin=False, register=True, overwrite_registry=True):

        """
        Args:
            name (str): unique name for the model
            connections (`list` of `dict`): list of connections dictionaries, which take
                the form ``{"inputs": [variables], "outputs": [variables]}``, e.g.:
                ``connections = [{"inputs": ["p", "T"], "outputs": ["V"]},``
                ``{"inputs": ["T", "V"], "outputs": ["p"]}]``
            constraints (str, Constraint, None): optional, string expressions or
                Constraint objects of some condition on which the model is
                valid, e. g. ``"n > 0"``, note that this must include variables if
                there is a ``variable_symbol_map``. Default: ``None`` (no constraints)
            display_names (`list` of `str`, `None`): optional, list of formatted names to use for
                display. Default: ``None`` (sets ``display_name`` property to a list with one item,
                which is the name in title case with underscores replaced with spaces,
                e.g. "name_of_a_model" would become "Name of a Model")
            description (str, None): optional, long form description of the model
            categories (`list` of `str`, `None`): optional, list of categories applicable to
                the model
            references (`list` of `str`, `None`): list of the informational links
                explaining / supporting the model. These strings should be a BibTeX string or
                in the form ``"type:info"`` where ``type`` is one of "url", "doi", or "isbn"
                and ``info`` contains the relevant data. These types will be automatically converted
                to BibTeX strings by automatic lookup using ``propnet.core.utils.references_to_bib()``.
                Users should verify the accuracy of the generated BibTeX strings.
                Default: ``None`` (no references...please add them if contributing!)
            implemented_by (`list` of `str`, `None`): optional, list of authors of the model by their
                GitHub usernames. Default: ``None`` (no authors...please attribute authorship if
                contributing!)
            variable_symbol_map (dict, None): optional, mapping of variables used as inputs/outputs to symbol names
                (e.g. ``{"n": "index_of_refraction"}`` etc.). Default: ``None`` (all inputs/outputs are
                named exactly the same as they symbols they represent)
            units_for_evaluation (str, dict, None): optional, if specified, coerces the units of
                inputs prior to evaluation and outputs post-evaluation to the units
                specified. If ``units_for_evaluation = 'default'``, all inputs/outputs will be
                converted to the unit specified by their associated Symbol object. If
                ``units_for_evaluation`` is a variable-keyed dict, the inputs/outputs
                will be converted to the units specified as values in the dict. If a variable is
                missing, it will be converted to the unit of the associated Symbol object.
                Default: ``'default'`` (all inputs/outputs are converted to canonical units for evaluation)
            test_data (`list` of `dict`, `None`): optional, test data with
                which to evaluate the model to verify correct function. Format:
                ``{'input': {variable: value}, 'output': {variable: value}}`` where `value`
                can be a string with unit (``'1.0 kg'``), BaseQuantity object, or bare number.
                Bare numbers will be assumed to be the units specified by ``units_for_evaluation``.
                If ``units_for_evaluation`` is not specified, units will be assumed from the
                associated Symbol object. Default: ``None`` (no test data)
            is_builtin (bool): **This option not intended to be set by users.** ``True`` if the model
                is included in propnet's core model library.
                Default: ``False`` (model is not in core library)
            register (bool): ``True`` registers the model with the model registry named by
                ``self._registry_name``. ``False`` instantiates the object without registering it.
                Registration fails if the name exists in the registry and ``overwrite_registry=False``.
                Default: ``True`` (register the model)
            overwrite_registry (bool): ``True`` overwrites the model registry if a model with
                the same name exists. ``False`` throws an error if a model with the same name
                exists in the registry. Default: ``True`` (replace same-named models)
        """

        self.name = name
        self._connections = connections
        self.description = description
        if display_names:
            if isinstance(display_names, str):
                self.display_names = [display_names]
            else:
                self.display_names = display_names
        elif self.name:
            self.display_names = [self.name.replace("_", " ").title()]
        else:
            self.display_names = []
        if isinstance(categories, str):
            categories = [categories]
        self.categories = categories or []
        if isinstance(implemented_by, str):
            implemented_by = [implemented_by]
        self.implemented_by = implemented_by or []
        self.references = references_to_bib(references or [])
        self._is_builtin = is_builtin

        # TODO: Should probably make the variable_symbol_map always be keyed
        #       by a string and valued by a symbol object for consistency
        # TODO: The creation of the VSM is weirdly cyclic but it works. Is there
        #       a better way to do this?
        # variable symbol map initialized as symbol name->symbol name, then updated
        # with any customization of variable to symbol mapping
        self._variable_symbol_map = {k: k for k in self.all_input_variables | self.all_output_variables}
        if variable_symbol_map:
            model_variable_symbol_map = {v: s for v, s in variable_symbol_map.items()
                                         if v in self._variable_symbol_map}
            self._variable_symbol_map.update(model_variable_symbol_map)
        self._verify_symbols_are_registered()
        self._variable_symbol_map = {v: Registry("symbols").get(s) or s
                                     for v, s in self._variable_symbol_map.items()}

        if units_for_evaluation or 'empirical' in self.categories:
            self._variable_unit_map = {prop_name: Registry("units").get(prop_name)
                                       for prop_name in self.all_symbols}
            # Update with explicitly supplied units if specified
            if isinstance(units_for_evaluation, dict):
                units_for_evaluation = {k: ureg.Unit(v).format_babel() for k, v in units_for_evaluation.items()}
                self._variable_unit_map.update(self.map_variables_to_symbols(units_for_evaluation))
            self._variable_unit_map = self.map_symbols_to_variables(self._variable_unit_map)
        else:
            self._variable_unit_map = {}

        # Define constraints by constraint objects or invoke from strings
        constraints = constraints or []
        self.constraints = []
        for constraint in constraints:
            if isinstance(constraint, Constraint):
                self.constraints.append(constraint)
            else:
                self.constraints.append(Constraint(constraint,
                                                   variable_symbol_map=variable_symbol_map))

        # Ensures our test data is variable-keyed and in the correct format
        test_data = test_data or self.load_test_data()
        if test_data:
            test_data = self._clean_test_data(test_data)
        self._test_data = test_data

        if register:
            self.register(overwrite_registry=overwrite_registry)

    def register(self, overwrite_registry=True):
        """
        Registers the model with the appropriate model registry.

        Args:
            overwrite_registry (bool): If a model with the same name
                as the current is already registered, `True` will overwrite
                the old model with the current and `False` will raise a
                KeyError.

        Raises:
            KeyError: if `overwrite_registry=False` and a model with the same
                name is already registered, this error is raised.
        """
        if not overwrite_registry and self.name in Registry(self._registry_name).keys():
            raise KeyError("Model '{}' already exists in the registry '{}'".format(self.name,
                                                                                   self._registry_name))
        Registry(self._registry_name)[self.name] = self

    def unregister(self):
        """
        Removes the model from its registry.

        """
        Registry(self._registry_name).pop(self.name, None)

    @property
    def registered(self):
        """
        Indicates if a model is registered with the model registry.

        Returns:
            bool: ``True`` if the model is registered. ``False`` otherwise.

        """
        return self.name in Registry(self._registry_name)

    def _clean_test_data(self, test_data):
        """
        Coerces test data into a value-unit string format (e.g. "5.0 kg").

        Args:
            test_data (`list` of `dict`): structured test data (see ``__init__()``)

        Returns:
            `list` of `dict`: test data converted to the value-unit format
        """
        clean_test_data = []
        for io_data_set in test_data:
            clean_data_set = {}
            for io, data_set in io_data_set.items():
                variable_value = self.map_symbols_to_variables(data_set)
                for variable, value in variable_value.items():
                    symbol = self.variable_symbol_map[variable]
                    clean_value = (value, self._variable_unit_map.get(variable) or Registry('units').get(symbol))
                    if Registry("symbols")[symbol].category != 'object':
                        # If not an object, check to see if number is specified
                        # with units as a tuple/list or as a string. If so,
                        # convert it.
                        try:
                            q = None
                            if isinstance(value, (list, tuple)) and len(value) == 2:
                                q = ureg.Quantity(*value)
                            elif isinstance(value, str):
                                q = ureg.Quantity(value)
                            elif isinstance(value, (BaseQuantity, ureg.Quantity)):
                                q = value
                            if q is not None:
                                clean_value = (q.magnitude, q.units.format_babel() if q.units else None)
                        except TypeError:
                            pass
                    variable_value[variable] = clean_value
                clean_data_set[io] = variable_value
            clean_test_data.append(clean_data_set)
        return clean_test_data

    def _verify_symbols_are_registered(self):
        """
        Ensures that all Symbol names associated with this model are
        registered in the symbol registry.

        Raises:
            KeyError: if a symbol is not registered, this error is raised
        """
        for prop in self.all_symbols:
            if prop not in Registry("symbols"):
                raise KeyError("Symbol '{}' is not registered in "
                               "symbol registry in model '{}'.".format(prop, self.name))

    @property
    def is_builtin(self):
        """
        Indicates whether the model is in the propnet core library.

        Returns:
            bool: ``True`` if the model is in the core library, ``False``
                if it is a custom-created model
        """
        return self._is_builtin

    @property
    def connections(self):
        """`list` of `dict`: list of connections dictionaries, which take
        the form:

        >>> {"inputs": [variables], "outputs": [variables]}

        For example:

        >>> connections = [
        >>>     {"inputs": ["p", "T"], "outputs": ["V"]},
        >>>     {"inputs": ["T", "V"], "outputs": ["p"]}
        >>> ]

        """
        return self._connections

    @property
    def variable_unit_map(self):
        """dict: variables mapped to the units, as ``pint`` Unit objects,
        representing the units used for the variables in the model. If a
        variable does not have a unit, the value is ``None``.
        """
        return {k: ureg.Unit(v) if v is not None else None
                for k, v in self._variable_unit_map.items()}

    @property
    def variable_symbol_map(self):
        """dict: variables mapped to the Symbol types that those variables represent
        """
        return self._variable_symbol_map

    @abstractmethod
    def plug_in(self, variable_value_dict):
        """
        Plugs in a variable to quantity dictionary

        Args:
            variable_value_dict (dict): a mapping
                of variables to values to be substituted
                into the model to yield output

        Returns:
            dict: output variables with associated
                values generated from the input
        """
        return

    def map_symbols_to_variables(self, symbols):
        """
        Helper method to convert symbol-keyed dictionary or list to
        variable-keyed dictionary or list

        Args:
            symbols (list or dict): list of symbols or symbol-
                keyed dictionary

        Returns (list or dict):
            list of variables or variable-keyed dict
        """
        rev_map = {v: k for k, v in getattr(self, "variable_symbol_map", {}).items()}
        return remap(symbols, rev_map)

    def map_variables_to_symbols(self, variables):
        """
        Helper method to convert variable-keyed dictionary or list to
        symbol-keyed dictionary or list

        Args:
            variables (`list`, `dict`, `set`): list of variables or variable-keyed
                dictionary

        Returns:
            `list` or `dict` or `set: list of symbols or symbol-keyed dict
        """
        return remap(variables, getattr(self, "variable_symbol_map", {}))

    def _convert_inputs_for_plugin(self, inputs):
        converted_inputs = {}
        for var, quantity in inputs.items():
            converted_inputs[var] = quantity.value
            if self.variable_unit_map.get(var) is not None:
                # Units are being assumed by equation and we need to strip them
                # or pint might get angry if it has to add or subtract quantities
                # with unmatched dimensions
                converted_inputs[var] = quantity.to(self.variable_unit_map[var]).magnitude
        return converted_inputs

    def _convert_outputs_from_plugin(self, outputs):
        converted_outputs = {}
        for var, quantity in outputs.items():
            symbol = self._variable_symbol_map[var]
            unit = self.variable_unit_map.get(var) or Registry("units").get(symbol)
            if unit is None:
                converted_outputs[var] = quantity
            else:
                if isinstance(quantity, ureg.Quantity):
                    try:
                        converted_outputs[var] = quantity.to(unit)
                    except DimensionalityError:
                        # If the equation multiplies by constants with dimensions,
                        # we'll end up with an output with incorrect dimensions.
                        # This forces the unit conversion until we can fix inclusion of constants
                        # TODO: Fix when we add support for constants with dimensions
                        converted_outputs[var] = ureg.Quantity(quantity.magnitude,
                                                               units=unit)
                else:
                    converted_outputs[var] = ureg.Quantity(quantity, units=unit)
        return converted_outputs

    def evaluate(self, symbol_quantity_dict, allow_failure=True, raise_timeout_errors=True):
        """
        Given a set of symbol values, performs error checking to see
        if the corresponding input variable values represents a valid
        input set based on the self.connections() method. If so, returns
        a dictionary representing the value of plug_in() applied to the
        input. The dictionary contains a "successful" key
        representing if plug_in() was successful.

        The key distinction between evaluate() and plug_in() is symbols
        in, symbols out vs. variables in, variables out.  In addition,
        evaluate also handles any requisite unit mapping.

        Args:
            symbol_quantity_dict (dict): a mapping of
                symbol names (str) to quantities (BaseQuantity) to be substituted
            allow_failure (bool): whether or not to catch
                errors in model evaluation
            raise_timeout_errors (bool): True ignores the value of "allow_failure"
                and forces TimeoutError exceptions to be raised. This is so they
                can be caught and handled by Graph, as timeouts are implemented there
                and not in Model.

        Returns:
            dict: dictionary of output symbols with associated values
                generated from the input, along with "successful" if the
                substitution succeeds
        """
        # Remap symbols and units if symbol map isn't none

        variable_quantity_dict = self.map_symbols_to_variables(
            symbol_quantity_dict)

        input_variable_quantity_dict = {k: v for k, v in variable_quantity_dict.items()
                                        if not (k in self.constraint_variables
                                                and k not in self.all_input_variables)}

        for (k, v) in variable_quantity_dict.items():
            # replacing = self.variable_symbol_map.get(k, k)
            replacing = self.variable_symbol_map.get(k)
            # to_quantity() returns original object if it's already a BaseQuantity
            # unlike Quantity() which will return a deep copy
            variable_quantity_dict[k] = QuantityFactory.to_quantity(replacing, v)

        # TODO: Is it really necessary to strip these?
        # TODO: maybe this only applies to pymodels or things with objects?
        # strip units from input and keep for reassignment

        variable_value_dict = self._convert_inputs_for_plugin(variable_quantity_dict)

        contains_complex_input = any(NumQuantity.is_complex_type(v) for v in variable_value_dict.values())
        input_variable_value_dict = {k: variable_value_dict[k] for k in input_variable_quantity_dict.keys()}

        # Plug in and check constraints
        try:
            with PrintToLogger(level="DEBUG") as plog, \
                    np.errstate(all='log'), \
                    warnings.catch_warnings():
                # Catch numpy warnings and log them
                np.seterrcall(plog)
                # Catch pint warning caused by numpy
                warnings.simplefilter("ignore", category=UnitStrippedWarning)
                out: dict = self.plug_in(input_variable_value_dict)
        except Exception as err:
            # Because the Timeout() context raises a TimeoutError as a response to a
            # signal from the os, the exception is raised wherever the program is
            # at the time, which is usually in "plug_in". We want to optionally raise
            # TimeoutErrors so they are caught by Graph() or an error handler otherwise
            # outside of this class.
            if not allow_failure or (isinstance(err, TimeoutError) and raise_timeout_errors):
                raise err
            else:
                return {"successful": False,
                        "message": "{} evaluation failed: {}".format(self, err)}
        if not self.check_constraints({**variable_value_dict, **out}):
            return {"successful": False,
                    "message": "Constraints not satisfied"}

        out = self._convert_outputs_from_plugin(out)
        out = self.map_variables_to_symbols(out)

        symbol_unit_map = self.map_variables_to_symbols(self.variable_unit_map)
        for symbol, value in out.items():
            provenance = ProvenanceElement(
                model=self.name, inputs=list(input_variable_quantity_dict.values()),
                source="propnet")

            try:
                quantity = QuantityFactory.create_quantity(
                    symbol, value,
                    symbol_unit_map.get(symbol) or Registry("units").get(symbol),
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

        if self.display_names:
            return self.display_names[0]
        return self.name.replace('_', ' ').title()

    # Note: these are formulated in terms of symbols rather than variables
    @property
    def input_sets(self):
        """
        Returns (set): set of input symbol sets
        """
        return [set(self.map_variables_to_symbols(d['inputs']))
                for d in self.connections]

    @property
    def output_sets(self):
        """
        Returns (set): set of output symbol sets
        """
        return [set(self.map_variables_to_symbols(d['outputs']))
                for d in self.connections]

    @property
    def all_inputs(self):
        """
        Returns (set): set of all input symbols
        """
        return set(chain.from_iterable(self.input_sets))

    @property
    def all_outputs(self):
        """
        Returns (set): set of all output symbols
        """
        return set(chain.from_iterable(self.output_sets))

    @property
    def all_symbols(self):
        """
        Returns (set): set of all symbols
        """
        return self.all_inputs.union(self.all_outputs).union(getattr(self, 'constraint_symbols', set()))

    @property
    def input_variable_sets(self):
        """
        Returns (set): set of input variable sets
        """
        return [set(d['inputs'])
                for d in self.connections]

    @property
    def output_variable_sets(self):
        """
        Returns (set): set of output variable sets
        """
        return [set(d['outputs'])
                for d in self.connections]

    @property
    def all_input_variables(self):
        """
        Returns (set): set of all input variables
        """
        return set(chain.from_iterable(self.input_variable_sets))

    @property
    def all_output_variables(self):
        """
        Returns (set): set of all output variables
        """
        return set(chain.from_iterable(self.output_variable_sets))

    @property
    def all_variables(self):
        """
        Returns (set): set of all variables
        """
        return self.all_input_variables.union(self.all_output_variables,
                                              getattr(self, 'constraint_variables', set()))

    @property
    def evaluation_list(self):
        """
        Gets everything one needs to call the evaluate method, which
        is all of the input symbols and constraints

        Returns:
            `list` of `set`: list of input sets with constraint symbols included
        """
        return [list(inputs | self.constraint_symbols - outputs)
                for inputs, outputs in zip(self.input_sets, self.output_sets)]

    def test(self, inputs, outputs):
        """
        Runs a test of the model to determine whether its operation
        is consistent with the specified inputs and outputs

        Args:
            inputs (dict): set of input names to values
            outputs (dict): set of output names to values

        Returns:
             bool: True if test succeeds
        """
        evaluate_inputs = self.map_variables_to_symbols(inputs)
        for symbol, value in evaluate_inputs.items():
            magnitude, unit = value
            evaluate_inputs[symbol] = QuantityFactory.create_quantity(
                symbol, magnitude,
                units=unit)
        outputs_from_model = self.evaluate(evaluate_inputs, allow_failure=False)
        outputs_from_model = self.map_symbols_to_variables(outputs_from_model)
        errmsg = "{} model test failed on ".format(self.name) + "{}\n"
        errmsg += "{}(test data) = {}\n"
        errmsg += "{}(model output) = {}"
        for var, known_output in outputs.items():
            symbol = self.variable_symbol_map[var]
            if isinstance(known_output, BaseQuantity):
                known_quantity = known_output
            else:
                # It's a tuple from read-in test data
                magnitude, units = known_output
                known_quantity = QuantityFactory.create_quantity(
                    symbol, magnitude,
                    units=units)
            evaluate_output = outputs_from_model[var]
            if not known_quantity.has_eq_value_to(evaluate_output):
                errmsg = errmsg.format("evaluate", var, evaluate_output,
                                       var, known_quantity)
                raise ModelEvaluationError(errmsg)

        return True

    def validate_from_preset_test(self):
        """
        Validates from test data based on the model name

        Returns:
            bool: True if validation completes successfully
        """
        if self._test_data is None:
            return False

        for test_dataset in self._test_data:
            self.test(**test_dataset)
        return True

    @property
    def constraint_symbols(self):
        """
        Returns:
            set: set of constraint input symbols
        """
        # Constraints are defined only in terms of symbols

        return set(self.map_variables_to_symbols(self.constraint_variables))

    @property
    def constraint_variables(self):
        """
        Returns:
            set: set of variables which are constrained
        """
        # Constraints are defined only in terms of variables
        all_vars = [c.all_inputs
                    for c in self.constraints]
        return set(chain.from_iterable(all_vars))

    def check_constraints(self, input_symbols):
        """
        Checks the constraints based on input symbol set

        Args:
            input_symbols (dict): symbol-value
                dictionary for input to constraints

        Returns:
            bool: True if constraints are satisfied, false if not
        """
        input_vars = self.map_symbols_to_variables(input_symbols)
        for constraint in self.constraints:
            if not constraint.plug_in(input_vars):
                return False
        return True

    # TODO: Can we roll this into the model module itself rather than hard coding
    #       this file path and loading from there?
    def load_test_data(self, test_data_path=None, deserialize=True):
        """
        Loads test data from preset or specified directory.
        Finds a json or yaml file with the prefix "name" and
        loads it.

        Args:
            test_data_path (str): test data file location
            deserialize (bool): whether or not to deserialize the test
                data, primarily used for printing example code

        Returns:
            dict: Dictionary of test data
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

        Returns:
            str: example code for this model

        """
        if self._test_data is None:
            return ""
        example_inputs = self._test_data[0]['inputs']
        # Strip units from outputs
        example_outputs = str({k: v[0] for k, v in self._test_data[0]['outputs'].items()})

        variable_definitions = []
        evaluate_args = []
        imports = []
        for input_name, input_value_and_unit in example_inputs.items():
            # Strip units from inputs
            input_value, _ = input_value_and_unit
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
            variable_str = "{input_name} = {input_value}".format(
                input_name=input_name,
                input_value=input_value_string,
            )
            variable_definitions.append(variable_str)

            evaluate_str = "\t'{}': {},".format(input_name, input_name)
            evaluate_args.append(evaluate_str)

        variable_definitions = '\n'.join(variable_definitions)
        evaluate_args = '\n'.join(evaluate_args)

        example_code = CODE_EXAMPLE_TEMPLATE.format(
            model_name=self.name,
            imports='\n'.join(imports),
            variable_definitions=variable_definitions,
            evaluate_args=evaluate_args,
            example_outputs=example_outputs)

        return example_code

    def __str__(self):
        return "Model: {}".format(self.name)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, rhs):
        if isinstance(rhs, Model):
            return self.name == rhs.name
        elif isinstance(rhs, str):
            return self.name == rhs
        return NotImplemented


CODE_EXAMPLE_TEMPLATE = """
from propnet.models import {model_name}
{imports}

{variable_definitions}

{model_name}.plug_in({{
{evaluate_args}
}})
\"\"\"
returns {example_outputs}
\"\"\"
"""


class EquationModel(Model, MSONable):
    """
    For models which are simple equations, use the ``EquationModel`` class. To construct a relationship between
    some property `symbol_a` and some other property `symbol_b`, both of which are registered in the propnet
    Registry, you can do the following:

    >>> from propnet.core.models import EquationModel
    >>> em = EquationModel('my_model', ['symbol_b = 2*symbol_a**3'])
    >>> em.plug_in({'symbol_a': 5})
    {'symbol_b': 250}

    Equations must be strings parsable by the ``sympy`` package. Note that the variable names in the equation are
    the same names as the symbols. If you would rather use a symbolic representation of the symbol names,
    specify the ``variable_symbol_map`` upon creation:

    >>> from propnet.core.quantity import QuantityFactory as QF
    >>> em = EquationModel('my_model', ['B = 2*A**3'],
    >>>                    variable_symbol_map={'A': 'symbol_a', 'B': 'symbol_b'})
    >>> em.plug_in({'A': 5})
    {'B': 250}

    """
    def __init__(self, name, equations, connections=None, constraints=None,
                 variable_symbol_map=None, display_names=None, description=None,
                 categories=None, references=None, implemented_by=None,
                 units_for_evaluation=None, solve_for_all_variables=False, test_data=None,
                 is_builtin=False, register=True, overwrite_registry=True):

        """
        Instantiates an equation-based model.

        Args:
            name (str): title of the model
            equations (`list` of `str`): list of string equations to parse
            connections (`list` of `dict`): list of connections dictionaries,
                which take the form {"inputs": [variables], "outputs": [variables]},
                for example:
                connections = [{"inputs": ["p", "T"], "outputs": ["V"]},
                               {"inputs": ["T", "V"], "outputs": ["p"]}]
                If no connections are specified (as default), EquationModel
                attempts to find them by the convention that the variable
                in front of the equals sign is the output and the variables
                to the right are the inputs, e. g. OUTPUT = INPUT_1 + INPUT_2,
                alternatively, using solve_for_all_variables will derive all
                possible input-output connections
            constraints (`list` of `str`, `list` of `Constraint`): constraints on models
            display_names (`list` of `str`): optional, list of alternative names to use for
                display
            description (str): long form description of the model
            categories (str): list of categories applicable to
                the model
            references (`list` of `str`): list of the informational links
                explaining / supporting the model. See ``Model.__init__()`` for specifics.
            test_data (`list` of `dict`): test data with
                which to evaluate the model. See ``Model.__init__()`` for specifics.
            is_builtin: See ``Model.__init__()``
            register: See ``Model.__init__()``
            overwrite_registry: See ``Model.__init__()``
        """

        self.equations = equations
        sympy_expressions = [parse_expr(eq.replace('=', '-(')+')')
                             for eq in equations]
        # If no connections specified, derive connections
        if connections is None:
            connections = []
            if solve_for_all_variables:
                connections, equations = [], []
                for expr in sympy_expressions:
                    for var in expr.free_symbols:
                        new = sp.solve(expr, var)
                        inputs = get_vars_from_expression(new)
                        connections.append(
                            {"inputs": inputs,
                             "outputs": [str(var)],
                             "_sympy_exprs": {str(var): new}
                             })
            else:
                for eqn in equations:
                    output_expr, input_expr = eqn.split('=')
                    inputs = get_vars_from_expression(input_expr)
                    outputs = get_vars_from_expression(output_expr)
                    connections.append(
                        {"inputs": inputs,
                         "outputs": outputs,
                         "_sympy_exprs": {outputs[0]: parse_expr(input_expr)}
                         })
        else:
            # TODO: I don't think this needs to be supported necessarily
            #       but it's causing problems with models with one input
            #       and two outputs where you only want one connection
            for connection in connections:
                new = sp.solve(sympy_expressions, connection['outputs'])
                sympy_exprs = {str(sym): solved
                               for sym, solved in new.items()}
                connection["_sympy_exprs"] = sympy_exprs

        super(EquationModel, self).__init__(
            name, connections, constraints, display_names, description,
            categories, references, implemented_by,
            variable_symbol_map, units_for_evaluation,
            test_data=test_data,
            is_builtin=is_builtin,
            register=register,
            overwrite_registry=overwrite_registry)

        self._generate_lambdas()

    def as_dict(self):
        d = {k if not k.startswith("_") else k.split('_', 1)[1]: v
             for k, v in self.__getstate__().items()}
        d['units_for_evaluation'] = d.pop('variable_unit_map')
        del d['is_builtin']
        for connection in d['connections']:
            del connection['_sympy_exprs']
        return d

    @classmethod
    def from_dict(cls, d, **kwargs):
        d_in = d.copy()
        d_in.update(kwargs)
        return cls(**d_in)

    @property
    def connections(self):
        if not all(['_lambdas' in connection.keys() for connection in self._connections]):
            self._generate_lambdas()
        return self._connections

    def _generate_lambdas(self):
        for connection in self._connections:
            for output_var, sympy_expr in connection['_sympy_exprs'].items():
                sp_lambda = sp.lambdify(connection['inputs'], sympy_expr)
                if '_lambdas' not in connection.keys():
                    connection['_lambdas'] = dict()
                connection['_lambdas'][output_var] = sp_lambda

    def __getstate__(self):
        d = deepcopy(self.__dict__)
        for connection in d['_connections']:
            if '_lambdas' in connection.keys():
                del connection['_lambdas']
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._generate_lambdas()

    def _convert_outputs_from_plugin(self, outputs):
        converted_outputs = {}
        for var, quantity in outputs.items():
            symbol = self._variable_symbol_map[var]
            unit = self.variable_unit_map.get(var) or Registry("units").get(symbol)
            if unit is None:
                raise ValueError("Unit for '{}' symbol is not specified by "
                                 "the model or in the registry".format(symbol))
            if isinstance(quantity, ureg.Quantity):
                try:
                    converted_outputs[var] = quantity.to(unit)
                except DimensionalityError:
                    # If the equation multiplies by constants with dimensions,
                    # we'll end up with an output with incorrect dimensions.
                    # This forces the unit conversion until we can fix inclusion of constants
                    # TODO: Fix when we add support for constants with dimensions
                    converted_outputs[var] = ureg.Quantity(quantity.magnitude,
                                                           units=unit)
            else:
                converted_outputs[var] = ureg.Quantity(quantity, units=unit)
        return converted_outputs

    def plug_in(self, variable_value_dict):
        """
        Equation plug-in solves the equation for all input
        and output combinations, returning the corresponding
        output values

        Args:
            variable_value_dict (dict): variable-keyed
                dict of values to be substituted

        Returns:
            dict: variable-keyed output dictionary
        """

        output = {}
        for connection in self.connections:
            if set(connection['inputs']) == set(variable_value_dict.keys()):
                for output_var, func in connection['_lambdas'].items():
                    output_vals = func(**variable_value_dict)
                    # TODO: this decision to only take max real values should
                    #       should probably be reevaluated at some point
                    # Scrub nan values and take max
                    if isinstance(output_vals, list):
                        try:
                            output_val = max([v for v in output_vals
                                              if not isinstance(v, complex)])
                        except ValueError:
                            raise ValueError("No real roots found for model {}".format(self.name))
                    else:
                        output_val = output_vals
                    output.update({output_var: output_val})
        if not output:
            raise ValueError("No valid input set found in connections")
        else:
            return output

    @classmethod
    def from_file(cls, filename, is_builtin=False, register=True, overwrite_registry=True):
        """
        Invokes EquationModel from filename

        Args:
            filename (str): filename containing model
            is_builtin (bool): See ``Model.__init__()``
            register (bool): See ``Model.__init__()``
            overwrite_registry (bool): See ``Model.__init__()``

        Returns:
            EquationModel: model corresponding to contents of file
        """
        model = loadfn(filename)
        if isinstance(model, dict):
            model['is_builtin'] = is_builtin
            model['register'] = register
            model['overwrite_registry'] = overwrite_registry
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
                 display_names=None, description=None, categories=None,
                 references=None, implemented_by=None, variable_symbol_map=None,
                 units_for_evaluation=True, test_data=None, is_builtin=False,
                 register=True, overwrite_registry=True):
        self._plug_in = plug_in
        super(PyModel, self).__init__(
            name, connections, constraints, display_names, description,
            categories, references, implemented_by,
            variable_symbol_map, units_for_evaluation,
            test_data=test_data, is_builtin=is_builtin,
            register=register,
            overwrite_registry=overwrite_registry)

    def plug_in(self, variable_value_dict):
        """
        plug_in for PyModel uses the attached _plug_in attribute
        as a method with the input variable_value_dict

        Args:
            variable_value_dict ({variable: value}): dict containing
                variable-keyed values to substitute

        Returns:
            dict: value of substituted expression
        """
        return self._plug_in(variable_value_dict)


# Note that this class exists purely as a factory method for PyModel
# which could be implemented as a class method of PyModel
# but wouldn't serialize as cleanly
class PyModuleModel(PyModel):
    """
    PyModuleModel is a class instantiated by a model path only,
    which exists primarily for the purpose of serializing python models
    """
    def __init__(self, module_path, is_builtin=False, register=True, overwrite_registry=True):
        """
        Args:
            module_path (str): path to module to instantiate model
        """
        self._module_path = module_path
        mod = __import__(module_path, globals(), locals(), ['config'], 0)
        super(PyModuleModel, self).__init__(**mod.config,
                                            is_builtin=is_builtin,
                                            register=register,
                                            overwrite_registry=overwrite_registry)

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
    Model based on deriving emerging properties (symbols) from collections of
    materials of known properties (symbols).

    Model requires the unambiguous assignment of materials to labels.
    These labeled materials' properties (symbols) are then referenced.
    """

    _registry_name = "composite_models"

    def __init__(self, name, connections, plug_in, pre_filter=None,
                 filter=None, **kwargs):
        """
        Args:
            name (str): title of the model
            connections (`list` of `dict`): list of connections dictionaries,
                which take the form {"inputs": [variables],
                                     "outputs": [variables]}, e. g.:
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

    def _verify_symbols_are_registered(self):
        for composite_prop in self.all_symbols:
            split_prop = composite_prop.rsplit('.', maxsplit=1)
            if len(split_prop) == 2:
                material_type, prop = split_prop
            else:
                material_type = ""
                prop = split_prop[0]
            if prop not in Registry("symbols").keys():
                raise KeyError("Symbol '{}' of material '{}' is not registered in "
                               "symbol registry in model '{}'.".format(composite_prop,
                                                                       material_type,
                                                                       self.name))

    @staticmethod
    def get_material(input_):
        """
        Args:
            input_ (String): inputs entry from the connections instance variable.
        Returns:
            String or None only material identifiers from the input argument.
        """
        separation = input_.split('.')
        components = len(separation)
        if components == 2:
            return separation[0]
        elif components > 2:
            raise Exception('Connections can only contain 1 period separator.')
        return None

    @staticmethod
    def get_variable(input_value):
        """
        Args:
            input_value (String): inputs entry from the connections.
        Returns:
            String only variable identifiers from the input argument.
        """
        separation = input_value.split('.')
        components = len(separation)
        if components == 1:
            return input_value
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
    def __init__(self, module_path, is_builtin=False, register=True, overwrite_registry=True):
        """
        Args:
            module_path (str): path to module to instantiate model
        """
        self._module_path = module_path
        mod = __import__(module_path, globals(), locals(), ['config'], 0)
        super(PyModuleCompositeModel, self).__init__(**mod.config,
                                                     is_builtin=is_builtin,
                                                     register=register,
                                                     overwrite_registry=overwrite_registry)

    def as_dict(self):
        return {"module_path": self._module_path,
                "@module": "propnet.core.model",
                "@class": "PyModuleCompositeModel"}


# Right now I don't see much of a use case for pythonic functionality
# here but maybe there should be
# TODO: Have this not inherit from Model because its functionality is
#       fundamentally different
class Constraint(Model):
    """
    Constraint class, resembles a model, but should outputs
    true or false based on a string expression containing
    input variables
    """
    def __init__(self, expression, name=None, **kwargs):
        """
        Args:
            expression (str): str to be parsed to evaluate constraint
            name (str): optional name for constraint, default None
            **kwargs: kwargs for model
        """
        self.expression = expression.replace(' ', '')
        # Parse all the non-math variables and assign to inputs
        split = re.split("[+-/*<>=()]", self.expression)
        inputs = [s for s in split if not will_it_float(s) and s]
        connections = [{"inputs": inputs, "outputs": ["is_valid"]}]
        Model.__init__(
            self, name=name, connections=connections, **kwargs)

    def _verify_symbols_are_registered(self):
        # Is it possible to not have any outputs for these models instead of "is_valid"?
        # If so, then we don't have to override this function
        for prop in self.all_inputs:
            if prop not in Registry("symbols").keys():
                raise KeyError("Symbol '{}' is not registered in "
                               "symbol registry for constraint '{}'.".format(prop, self.expression))

    def register(self, overwrite_registry=True):
        pass

    def plug_in(self, variable_value_dict):
        """
        Evaluates the expression with sympy and provided values
        and returns the boolean of that expression

        Args:
            variable_value_dict (dict): dict containing
                variable-keyed values to substitute

        Returns:
            dict: value of substituted expression
        """
        return parse_expr(self.expression, variable_value_dict)

    def __repr__(self):
        return "Constraint: {}".format(self.expression)

    def __hash__(self):
        return hash(self.expression)

    def __eq__(self, rhs):
        if isinstance(rhs, Model):
            return self.expression == rhs.expression
        elif isinstance(rhs, str):
            return self.expression == rhs
        return NotImplemented


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
    based on an input map, used to translate variables to symbols
    and vice-versa

    Args:
        obj (`list`, `dict`, `set`): an iterable of symbols or symbol-keyed
            dictionary to be remapped using variables.
        mapping (dict): dictionary of values to remap

    Returns:
        `list`, `dict`, `set`: remapped list of items or item-keyed dictionary
    """
    if isinstance(obj, dict):
        new = {mapping.get(in_key) or in_key: obj.get(in_key)
               for in_key in obj.keys()}
    else:
        new = [mapping.get(in_item) or in_item for in_item in obj]
        if isinstance(obj, set):
            new = set(new)
    return new


def get_vars_from_expression(expression):
    """
    Helper function to get all sympy symbols (vars) from a string expression

    Args:
        expression (str or sympy expression): string or sympy expression
    """
    if isinstance(expression, six.string_types):
        expression = parse_expr(expression)
    if isinstance(expression, list):
        out = list(chain.from_iterable([get_vars_from_expression(expr)
                                        for expr in expression]))
    else:
        out = [str(v) for v in expression.free_symbols]
    return list(set(out))
