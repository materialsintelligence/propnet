"""This module defines classes related to Symbol descriptions"""

import six
import numpy as np

from monty.json import MSONable
from ruamel.yaml import safe_dump

from propnet import logger, ureg


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
                 category='property'):
        """
        Parses and validates a series of inputs into a PropertyMetadata
        tuple, a format that PropNet expects.

        Parameters correspond exactly with those of a PropertyMetadata tuple.

        Args:
            name (str): string ASCII identifying the property uniquely
                as an internal identifier.
            units (str or tuple): units of the property as a Quantity
                supported by the Pint package.  Can be supplied as a
                string (e. g. cm^2) or a tuple for Quantity.from_tuple
                (e. g. [1.0, [['centimeter', 1.0]]])
            display_names (list<str>): list of strings giving possible
                human-readable names for the property.
            display_symbols (list<str>): list of strings giving possible
                human-readable symbols for the property.
            shape (id): list giving the order of the tensor as the length,
                and number of dimensions as individual integers in the list.
            comment (str): any useful information on the property including
                its definitions and possible citations.
            category (str): 'property', for property of a material,
                            'condition', for other variables,
                            'object', for a value that is a python object.
            object_type (class): class representing the object stored in these symbols.
        """

        # TODO: not sure object should be distinguished
        # TODO: clean up for divergent logic
        if category not in ('property', 'condition', 'object'):
            raise ValueError('Unsupported category: {}'.format(category))

        if not name.isidentifier():
            raise ValueError(
                "The canonical name ({}) is not valid.".format(name))

        if display_names is None or len(display_names) == 0:
            raise ValueError(
                "Insufficient display names for ({}).".format(name))

        if category in ('property', 'condition'):

            if object_type is not None:
                raise ValueError(
                    "Cannot define an object type for a {}.".format(category))

            try:
                np.zeros(shape)
            except TypeError:
                raise TypeError(
                    "Shape provided for ({}) is invalid.".format(id))

            logger.info("Units parsed from a string format automatically, "
                        "do these look correct? %s", units)
            if isinstance(units, six.string_types):
                units = 1 * ureg.parse_expression(units)
            else:
                units = ureg.Quantity.from_tuple(units)
        else:
            if units is not None:
                raise ValueError("Cannot define units for generic objects.")
            units = ureg.parse_expression("")  # dimensionless

        self.name = name
        self.category = category
        self.units = units
        self.object_type = object_type
        self.display_names = display_names
        self.display_symbols = display_symbols
        self.shape = shape
        self.comment = comment

    @property
    def dimension_as_string(self):
        """
        Returns:
            (str): shape of property (np.shape) as a human-readable string
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
    def compatible_units(self):
        """
        Returns: list of compatible units as strings
        """
        try:
            compatible_units = [str(u) for u in self.units.compatible_units()]
            return compatible_units
        except KeyError:
            logger.warning("Cannot find compatible units for %s", self.name)
            return []

    def __hash__(self):
        return self.name.__hash__()

    def __eq__(self, other):
        if isinstance(other, Symbol):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other

    def __str__(self):
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