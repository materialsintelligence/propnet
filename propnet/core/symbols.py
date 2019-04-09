"""This module defines classes related to Symbol descriptions"""

import six
import numpy as np
from copy import copy

from monty.json import MSONable
from ruamel.yaml import safe_dump

from propnet import logger, ureg
from propnet.core.registry import Registry
from sympy.parsing.sympy_parser import parse_expr
import sympy as sp

# TODO: This could be split into separate classes
#       or a base class + subclasses for symbols with
#       units vs those without
# TODO: also symbols with branch points for things like
#       sets of finite conditions (miller indices)
# TODO: I think object properties (e. g. structure)
#       should have a "unitizer" metaclass that unitizes
#       their attributes


class Symbol(MSONable):
    """
    Class storing the complete description of a Symbol.

    These classes are typically instantiated at runtime when all
    .yaml files are read in from the symbols folder.

    """

    def __init__(self, name, display_names=None, display_symbols=None,
                 units=None, shape=None, object_type=None, comment=None,
                 category='property', constraint=None, default_value=None,
                 is_builtin=False, register=True, overwrite_registry=True):
        """
        Instantiates Symbol object.

        Args:
            name (str): string ASCII identifying the property uniquely
                as an internal identifier.
            units (str, tuple): units of the property as a Quantity
                supported by the Pint package.  Can be supplied as a
                string (e. g. ``cm^2``) or a tuple for ``Quantity.from_tuple``
                (e. g. ``[1.0, [['centimeter', 1.0]]]``)
            display_names (`list` of `str`): list of strings giving possible
                human-readable names for the property.
            display_symbols (`list` of `str`): list of strings giving possible
                human-readable symbols for the property.
            shape (list, int): list giving the order of the tensor as the length,
                and number of dimensions as individual integers in the list.
                If an integer is provided, the symbol contains a vector. If ``shape=1``,
                the symbol contains a scalar.
            comment (str): any useful information on the property including
                its definitions and possible citations.
            category (str): 'property', for property of a material,
                            'condition', for other variables,
                            'object', for a value that is a python object.
            object_type (type, str): class or name of a class representing the object stored in
                these symbols.
            constraint (str): constraint associated with the symbol, must
                be a string expression (e. g. inequality) using the symbol
                name, e. g. ``bulk_modulus > 0``.
            default_value (any): default value for the symbol, e. g. 300 for
                temperature or 1 for magnetic permeability
            is_builtin (bool): True if the model is included with propnet
                by default. Not intended to be set explicitly by users
            register (bool): True if the model should be registered in the symbol registry
                upon instantiation
            overwrite_registry (bool): True if the value in the symbol registry should be
                overwritten if it exists. False will raise a KeyError if a symbol with the
                same name is already registered.

        """

        # TODO: not sure object should be distinguished
        # TODO: clean up for divergent logic
        if category not in ('property', 'condition', 'object'):
            raise ValueError('Unsupported category: {}'.format(category))

        if not name.isidentifier():
            raise ValueError(
                "The canonical name ({}) is not valid.".format(name))

        if not display_names:
            display_names = [name]

        self.object_type = None
        self._object_class = None
        self._object_module = None

        if category in ('property', 'condition'):

            if object_type is not None:
                raise ValueError(
                    "Cannot define an object type for a {}.".format(category))

            try:
                np.zeros(shape)
            except TypeError:
                raise TypeError(
                    "Shape provided for ({}) is invalid.".format(name))

            if units is None:
                units = 1 * ureg.dimensionless
            elif isinstance(units, six.string_types):
                units = 1 * ureg.parse_expression(units)
            elif isinstance(units, (tuple, list)):
                units = ureg.Quantity.from_tuple(units)
            else:
                raise TypeError("Cannot parse unit format: {}".format(units))

            units = units.units.format_babel()
        else:
            if units is not None:
                raise ValueError("Cannot define units for generic objects.")
            units = None # ureg.parse_expression("")  # dimensionless

            if object_type:
                if isinstance(object_type, type):
                    self._object_module = object_type.__module__
                    self._object_class = object_type.__name__
                else:
                    # Do not try to import the module for security reasons.
                    # We don't want malicious modules to be automatically imported.
                    modclass = object_type.rsplit('.', 1)
                    if len(modclass) == 1:
                        self._object_module = 'builtins'
                        self._object_class = modclass[0]
                    else:
                        self._object_module, self._object_class = modclass

                if self._object_module == 'builtins':
                    self.object_type = self._object_class
                else:
                    self.object_type = ".".join([self._object_module,
                                                 self._object_class])

        self.name = name
        self.category = category
        self._units = units
        self.display_names = display_names
        self.display_symbols = display_symbols
        # If a user enters [1] or [1, 1, ...] for shape, treat as a scalar
        if shape and np.size(np.zeros(shape=shape)) == 1:
            shape = 1
        # If a user enters a 0 dimension, throw an error
        if shape and np.size(np.zeros(shape=shape)) == 0:
            raise ValueError("Symbol cannot have a shape with a 0-size dimension: {}".format(shape))
        self.shape = shape
        self.comment = comment
        self.default_value = default_value
        self._is_builtin = is_builtin

        # TODO: This should explicity deal with only numerical symbols
        #       because it uses sympy to evaluate them until we make
        #       a class to evaluate them using either sympy or a custom func
        # Note that symbol constraints are not constraint objects
        # at the moment because using them would result in a circular

        if constraint is not None:
            try:
                func = sp.lambdify(self.name, parse_expr(constraint))
                func(0)
            except Exception:
                raise ValueError("Constraint expression invalid: {}".format(constraint))

        self._constraint = constraint
        self._constraint_func = None

        if register:
            self.register(overwrite_registry=overwrite_registry)

    def register(self, overwrite_registry=False):
        """
        Registers the symbol with the symbol registry.

        Args:
            overwrite_registry (bool): If a symbol with the same name
                as the current is already registered, `True` will overwrite
                the old symbol with the current and `False` will raise a
                KeyError.

        Raises:
            KeyError: if `overwrite_registry=False` and a symbol with the same
                name is already registered, this error is raised.

        """
        if not overwrite_registry and \
                (self.name in Registry("symbols").keys() or self.name in Registry("units").keys()):
            raise KeyError("Symbol '{}' already exists in the symbol or unit registry".format(self.name))

        Registry("symbols")[self.name] = self
        Registry("units")[self.name] = self.units.format_babel() if self.units else None
        if self.default_value is not None:
            Registry("symbol_values")[self.name] = self.default_value

    def unregister(self):
        """
        Removes the symbol from all applicable registries.

        """
        Registry("symbols").pop(self.name, None)
        Registry("units").pop(self.name, None)
        Registry("symbol_values").pop(self.name, None)

    @property
    def registered(self):
        """
        Indicates if a symbol is registered with the symbol registry.

        Returns:
            bool: True if the symbol is registered. False otherwise.

        """
        return self.name in Registry("symbols").keys()

    @property
    def constraint(self):
        """
        Gets callable constraint function for this symbol.

        Returns:
            callable: sympy lambda function representing the symbol constraint
        """
        if self._constraint:
            if self._constraint_func is None:
                self._constraint_func = sp.lambdify(self.name, parse_expr(self._constraint))
            return self._constraint_func
        return None

    def __getstate__(self):
        d = copy(self.__dict__)
        d['_constraint_func'] = None
        return d

    @property
    def is_builtin(self):
        """
        Indicates whether the symbol is a propnet built-in.

        Returns:
            bool: ``True`` if the symbol is a built-in, ``False``
                if it is a custom-created symbol
        """
        return self._is_builtin

    @property
    def units(self):
        return ureg.Unit(self._units) if self._units else None

    @property
    def object_class(self):
        return self._object_class

    @property
    def object_module(self):
        return self._object_module

    @property
    def dimension_as_string(self):
        """
        Produces the shape including form factor (scalar, vector, matrix, tensor)

        Returns:
            str: shape of property (np.shape) as a human-readable string
        """

        if isinstance(self.shape, int):
            return 'scalar'
        elif isinstance(self.shape, list) and len(self.shape) == 1:
            return '{} vector'.format(self.shape)
        elif isinstance(self.shape, list) and len(self.shape) == 2:
            return '{} matrix'.format(self.shape)
        else:
            # technically might not always be true
            return '{} tensor'.format(self.shape)

    @property
    def unit_as_string(self):
        """
        Produces units of the symbol as a human-readable string

        Returns:
            str: units
        """

        if self.units.dimensionless:
            return "dimensionless"

        # Below is a special formatting string specific to pint units
        unit_str = '{:~P}'.format(self.units)

        return unit_str

    @property
    def compatible_units(self):
        """
        Gets a list of units with compatible dimensionality to the symbol's unit

        Returns:
            `list` of `str`: compatible units
        """
        try:
            compatible_units = [str(u) for u in self.units.compatible_units()]
            return compatible_units
        except KeyError:
            logger.warning("Cannot find compatible units for %s", self.name)
            return []

    def is_correct_object_type(self, obj):
        if self.category == 'object':
            if not self.object_module and not self.object_class:
                # If no type was specified, just accept any object.
                # Leave it up to the model evaluation procedures to type check
                return True

            modname = obj.__class__.__module__
            clsname = obj.__class__.__name__

            return self.object_module == modname and self.object_class == clsname
        else:
            raise AttributeError("Object type not defined for symbol of category '{}'".format(self.category))

    def __hash__(self):
        return self.name.__hash__()

    def __eq__(self, other):
        if isinstance(other, Symbol):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        return NotImplemented

    def __str__(self):
        return "Symbol: {}".format(self.name)

    @property
    def summary(self):
        """
        Prints a full summary of the symbol
        """
        to_return = self.name + ":\n"
        for k, v in self.__dict__.items():
            to_return += "\t" + k + ":\t" + str(v) + "\n"
        return to_return

    def __repr__(self):
        return "{}<{}>".format(self.category, self.name)

    # TODO: I don't think this is necessary, double check
    def to_yaml(self):
        """
        Method to serialize the symbol in a yaml format
        """
        data = {
            "name": self.name,
            "category": self.category,
            "display_names": self.display_names,
            "display_symbols": self.display_symbols,
            "comment": self.comment
        }

        if self.units:
            data["units"] = self.units.to_tuple()
        if self.shape:
            data["shape"] = self.shape
        if self.object_type:
            data["object_type"] = self.object_type

        return safe_dump(data)

    def as_dict(self):
        d = {'@module': self.__module__,
             '@class': self.__class__.__name__,
             'name': self.name,
             'display_names': self.display_names,
             'display_symbols': self.display_symbols,
             'units': (1 * self.units).to_tuple() if self.units else None,
             'shape': self.shape,
             'object_type': self.object_type,
             'comment': self.comment,
             'category': self.category,
             'constraint': self.constraint,
             'default_value': self.default_value,
             'is_builtin': self._is_builtin}

        return d
