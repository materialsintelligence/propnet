import numpy as np

from typing import *
from propnet import logger, ureg
from pybtex.database.input.bibtex import Parser
from monty.json import MSONable

class SymbolMetadata(MSONable):
    """Class storing the complete description of a property."""

    __slots__ = ['name', 'units', 'display_names', 'display_symbols',
                 'dimension', 'test_value', 'comment', 'type']

    def __init__ (self, name: str, units: ureg.Quantity, display_names: List[str],
                  display_symbols: List[str], dimension: List[int], test_value: np.ndarray,
                  comment: str, type: str = 'property'):
        """
        Parses and validates a series of inputs into a PropertyMetadata tuple, a format that PropNet expects.
        Parameters correspond exactly with those of a PropertyMetadata tuple.
        :param name: string ASCII identifying the property uniquely as an internal identifier.
        :param units: units of the property as a Quantity supported by the Pint package.
        :param display_names: list of strings giving possible human-readable names for the property.
        :param display_symbols: list of strings giving possible human-readable symbols for the property.
        :param dimension: list giving the order of the tensor as the length, and number of dimensions as individual
                          integers in the list.
        :param test_value: a sample value of the property, reasonable over a wide variety of contexts.
        :param comment: any useful information on the property including its definitions and possible citations.
        :param type: 'property', if a property of a material, or 'condition' for other variables (e.g. temperature)
        :return: PropertyMetadata instance.
        """

        # TODO: need to formalize property vs condition distinction

        if type not in ('property', 'condition', 'object'):
            raise ValueError('Unsupported property type')

        if not name.isidentifier() or not name.islower():
            raise ValueError("The canonical name ({}) is not valid.".format(id))

        if display_names is None or len(display_names) == 0:
            raise ValueError("Insufficient display names for ({}).".format(id))


        if type in ('property', 'condition', 'objects'):

            # additional checking

            try:
                np.zeros(dimension)
            except:
                raise ValueError("Dimensions provided for ({}) are invalid.".format(id))

            if not np.any(test_value):
                logger.warn("Test value for {} is zero. "
                            "Please change to a more appropriate test value.".format(name))

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
        """:return: dimension of property (np.shape) as a human-readable string"""

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
        """ """

        # self.units has both the units and (sometimes) a
        # prefactor (its magnitude)
        unit_str = '{:~P}'.format(self.units.units)

        if self.units.magnitude != 1:
            unit_str = '{} {}'.format(self.units.magnitude, unit_str)

        return unit_str

    def __eq__(self, other):
        return self.name == other.name


class Symbol(MSONable):
    """Class storing the value of a property in a given context."""

    def __init__(self, type, value, tags,
                 provenance=None):
        """
        Parses inputs for constructing a Property object.
        :param type: pointer to an existing PropertyMetadata object,
        identifies the type of data stored in the property.
        :param value: value of the property
        :param comment: important information relative to the property, including sourcing.
        :param source_ids: the mpIDs from which the property value originates.
        """
        self._type = type
        self._value = value
        self._tags = tags
        self._provenance = provenance

    @property
    def value(self):
        return self._value

    @property
    def type(self):
        return self._type

    @property
    def tags(self):
        return self._tags

    @property
    def provenance(self):
        return self._provenance