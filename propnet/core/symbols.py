import numpy as np

from typing import *
from uuid import UUID
from monty.json import MSONable

from propnet import logger, ureg
from propnet.core.utils import uuid


class Symbol(MSONable):
    """
    Class storing the complete description of a Symbol.

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
        (str) comment -> (str) Gives any important commentary on the property.

    Attributes (see above for descriptions):
        name: (str)
        units: (list<id>)
        display_names: (list<str>)
        display_symbols: (list<str>)
        dimension: (id)
        comment: (str)
    """

    # TODO: this really needs to be rethought, possibly split into separate classes
    # or a base class + subclasses for symbols with units vs those without
    def __init__(self, name, display_names,
                 display_symbols, units=None, dimension=None,
                 object_type=None,
                 comment=None, category='property'):
        """
        Parses and validates a series of inputs into a PropertyMetadata tuple, a format that
        PropNet expects.
        Parameters correspond exactly with those of a PropertyMetadata tuple.

        Args:
            name (str): string ASCII identifying the property uniquely as an internal identifier.
            units (id): units of the property as a Quantity supported by the Pint package.
            display_names (list<str>): list of strings giving possible human-readable names for
            the property.
            display_symbols (list<str>): list of strings giving possible human-readable symbols
            for the property.
            dimension (id): list giving the order of the tensor as the length, and number of
            dimensions as individual
                            integers in the list.
            comment (str): any useful information on the property including its definitions and
            possible citations.
            category (str): 'property', if a property of a material, or 'condition' for other
            variables (e.g. temperature)
        """

        if category not in ('property', 'condition', 'object'):
            raise ValueError('Unsupported property category')

        if not name.isidentifier() or not name.islower():
            raise ValueError("The canonical name ({}) is not valid.".format(name))

        if display_names is None or len(display_names) == 0:
            raise ValueError("Insufficient display names for ({}).".format(name))

        if category in ('property', 'condition'):

            if object_type is not None:
                raise ValueError("Cannot define an object type for a {}.".format(category))

            try:
                np.zeros(dimension)
            except TypeError:
                raise TypeError("Dimensions provided for ({}) are invalid.".format(id))

            try:
                units = ureg.Quantity.from_tuple(units)
                # calling dimensionality explicitly checks units are defined in registry
                units.dimensionality
            except Exception as e:
                raise ValueError("Problem loading units for {}: {}".format(name, e))

        else:
            if units is not None:
                raise ValueError("Cannot define units for generic objects.")

        self.name = name
        self.category = category
        self.units = units
        self.object_type = object_type
        self.display_names = display_names
        self.display_symbols = display_symbols
        self.dimension = dimension  # TODO: rename to shape?
        self.comment = comment

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
    def unit_as_string(self) -> str:
        """
        Returns: unit of property as human-readable string
        """

        if self.units.dimensionless:
            return "dimensionless"

        # self.units has both the units and (sometimes) a
        # prefactor (its magnitude)
        unit_str = '{:~P}'.format(self.units.units)

        if self.units.magnitude != 1:
            unit_str = '{} {}'.format(self.units.magnitude, unit_str)

        return unit_str

    @property
    def compatible_units(self) -> List[str]:
        """
        Returns: list of compatible units as strings
        """
        try:
            compatible_units = [str(unit) for unit in self.units.compatible_units()]
            return compatible_units
        except KeyError:
            logger.warn("Could not find compatible units for {}".format(self.name))
            return []

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        to_return = self.name + ":\n"
        for item in self.__slots__:
            to_return += "\t" + item + ":\t" + str(self.__getattribute__(item)) + "\n"
        return to_return

    def __repr__(self):
        return "{}<{}>".format(self.category, self.name)

    def __hash__(self):
        return self.name.__hash__()