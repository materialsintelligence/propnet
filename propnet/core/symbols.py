import numpy as np
import sys

from typing import *
from propnet import logger, ureg
from pybtex.database.input.bibtex import Parser
from monty.json import MSONable


class SymbolMetadata(MSONable):
    """
    Class storing the complete description of a SymbolType.

    These classes are instantiated at runtime when all .yaml files are read in from the symbols folder. Such .yaml files
    should contain the following fields:
        (str) name -> (str) Internal name of the property, matches .yaml file name, must be a valid Python variable name
        (str) units -> (list<id>) First element is always the float 1.0. Proceeding elements are lists of lists.
                                  The inner list contains a base SI unit str and a float representing that base unit
                                  raised to that power.
                                  Multiple inner lists can occur in the outer list, representing multiplication of
                                  units together.
        (str) display_names -> (list<str>) list of human-readable strings describing the property.
        (str) display_symbols -> (list<str>) list of latex-able strings producing symbols associated with representing
                                             this property in equations, etc.
        (str) dimension -> (id) gives the length dimensions of the n-dimensional array required to represent the property.
                                This value is a list if multiple length dimensions are required or an integer if only
                                one length dimension is required.
        (str) test_value -> (id) n dimensional array giving a sample value of the property.
        (str) comment -> (str) Gives any important commentary on the property.

    Attributes (see above for descriptions):
        name: (str)
        units: (list<id>)
        display_names: (list<str>)
        display_symbols: (list<str>)
        dimension: (id)
        test_value: (id)
        comment: (str)
    """

    __slots__ = ['name', 'units', 'display_names', 'display_symbols',
                 'dimension', 'test_value', 'comment', 'type']

    def __init__(self, name: str, units: ureg.Quantity, display_names: List[str],
                 display_symbols: List[str], dimension: List[int], test_value: np.ndarray,
                 comment: str, type: str = 'property', strict: bool = True):
        """
        Parses and validates a series of inputs into a PropertyMetadata tuple, a format that PropNet expects.
        Parameters correspond exactly with those of a PropertyMetadata tuple.

        Args:
            name (str): string ASCII identifying the property uniquely as an internal identifier.
            units (id): units of the property as a Quantity supported by the Pint package.
            display_names (list<str>): list of strings giving possible human-readable names for the property.
            display_symbols (list<str>): list of strings giving possible human-readable symbols for the property.
            dimension (id): list giving the order of the tensor as the length, and number of dimensions as individual
                            integers in the list.
            test_value (id): a sample value of the property, reasonable over a wide variety of contexts.
            comment (str): any useful information on the property including its definitions and possible citations.
            type (str): 'property', if a property of a material, or 'condition' for other variables (e.g. temperature)
            strict (bool): flag indicating if error checking should occur on the inputs.

        Returns:
            (PropertyMetadata) PropertyMetadata instance.
        """

        if strict:

            if type not in ('property', 'condition', 'object'):
                raise ValueError('Unsupported property type')

            if not name.isidentifier() or not name.islower():
                raise ValueError("The canonical name ({}) is not valid.".format(name))

            if display_names is None or len(display_names) == 0:
                raise ValueError("Insufficient display names for ({}).".format(name))

            if type in ('property', 'condition'):

                # additional checking

                try:
                    np.zeros(dimension)
                except TypeError:
                    raise TypeError("Dimensions provided for ({}) are invalid.".format(id))

                try:
                    units = ureg.Quantity.from_tuple(units)
                    # calling dimensionality explicitly checks units are defined in registry
                    units.dimensionality
                except Exception as e:
                    raise ValueError('Problem loading units for {}: {}'.format(name, e))

        self.name = name
        self.units = units
        self.display_names = display_names
        self.display_symbols = display_symbols
        self.dimension = dimension  # TODO: rename to shape?
        self.test_value = test_value
        self.comment = comment
        self.type = type

    @property
    def dimension_as_string(self):
        """
        Returns:
            (str): dimension of property (np.shape) as a human-readable string
        """

        if isinstance(self.dimension, int):
            return 'scalar'
        elif isinstance(self.dimension, list) and len(self.dimension) == 1:
            return '{} vector'.format(self.dimension)
        elif isinstance(self.dimension, list) and len(self.dimension) == 2:
            return '{} matrix'.format(self.dimension)
        else:
            # technically might not always be true
            return '{} tensor'.format(self.dimension)

    @property
    def unit_as_string(self):
        """
        Returns:
            (str): unit of property as human-readable string
        """

        # self.units has both the units and (sometimes) a
        # prefactor (its magnitude)
        unit_str = '{:~P}'.format(self.units.units)

        if self.units.magnitude != 1:
            unit_str = '{} {}'.format(self.units.magnitude, unit_str)

        return unit_str

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class Symbol(MSONable):
    """
    Class storing the value of a property.

    Constructed by the user to assign values to abstract SymbolMetadata types. Represents the fact that a given Symbol
    has a given value. They are added to the PropertyNetwork graph in the context of Material objects that store
    collections of Symbol objects representing that a given material has those properties.

    Attributes:
        type: (SymbolMetadata) the type of information that is represented by the associated value.
        value: (id) the value associated with this symbol.
        tags: (list<str>)
    """

    def __init__(self, type, value, tags,
                 provenance=None):
        """
        Parses inputs for constructing a Property object.

        Args:
            type (SymbolMetadata): pointer to an existing PropertyMetadata object, identifies the type of data stored
                                   in the property.
            value (id): value of the property.
            tags (list<str>): list of strings storing metadata from Symbol evaluation.
            provenance (id): time of creation of the object.
        """
        self._type = type
        self._value = value
        self._tags = tags
        self._provenance = provenance

    # Associated accessor methods.
    @property
    def value(self):
        """
        Returns:
            (id): value of the Symbol
        """
        return self._value

    @property
    def type(self):
        """
        Returns:
            (SymbolMetadata): SymbolMetadata of the Symbol
        """
        return self._type

    @property
    def tags(self):
        """
        Returns:
            (list<str>): tags of the Symbol
        """
        return self._tags

    @property
    def provenance(self):
        """
        Returns:
            (id): time of creation of the Symbol
        """
        return self._provenance

    def __hash__(self):
        return hash(self.type.name)

    def __eq__(self, other):
        if not isinstance(other, Symbol):
            return False
        if self.type != other.type:
            return False
        if not np.isclose(float(self.value), float(other.value)):
            return False
        return True
