import numpy as np

from typing import List, Dict, SupportsFloat, NamedTuple, Optional, Union
from propnet import logger, ureg
from glob import glob
from enum import Enum
from monty.serialization import loadfn
from pybtex.database.input.bibtex import Parser

PropertyType = NamedTuple('PropertyType', [('name', str),
                                           ('units', ureg.Quantity),
                                           ('display_names', List[str]),
                                           ('display_symbols', List[str]),
                                           ('dimension', List),
                                           ('test_value', np.ndarray),
                                           ('comment', str)])

class PropertyLoader():

    def __init__(self, properties: List[Dict]):
        """

        :param properties:
        """

        self.properties = [self.parse_property(property)
                           for property in properties]


    @classmethod
    def from_path(cls, path: str):
        """

        :param path:
        :return:
        """

        files = glob("{}/*.yaml".format(path))
        properties = [loadfn(f) for f in files]

        return cls(properties)

    def parse_property(self, property: dict) -> PropertyType:
        """

        :param property:
        :return:
        """

        name = property['name']

        # check canonical name
        if not name.isidentifier() or \
           not name.islower():
            raise ValueError("The canonical name ({}) is not valid."
                             .format(name))

        # check units supported by pint
        units = ureg.Quantity.from_tuple((1, tuple(tuple(x) for x in property['units'])))
        property['units'] = units

        # check required fields
        if len(property['display_names']) == 0:
            raise ValueError("Please provide at least one display name for {}."
                             .format(name))
        if len(property['display_symbols']) == 0:
            raise ValueError("Please provide at least one display symbol for {}."
                             .format(name))

        # check test value is valid
        test_value = float(property['test_value'])
        if test_value == 0:
            logger.warn("Test value for {} is 0, is there a more appropriate test value?"
                        .format(name))

        # check dimensions are valid
        try:
            empty_array = np.zeros(property['dimension'])
        except:
            raise ValueError("Invalid dimensions for {}.".format(name))

        return PropertyType(**property)


class PropertyQuantity:

    def __init__(self,
                 name: str,
                 quantity: ureg.Quantity,
                 sources: Optional[List] = None,
                 references: Optional[List[str]] = None,
                 assumptions: Optional[List] = None):

        self.name = name
        self.quantity = quantity
        self.sources = sources

        parser = Parser()
        self.references = [parser.parse_string(s) for s in references]

        self.assumptions = assumptions


    @classmethod
    def with_unit_string(cls,
                         name: str,
                         value: np.ndarray,
                         units: str,
                         sources: Optional[List] = None,
                         references: Optional[List[str]] = None,
                         assumptions: Optional[List] = None):

        units = ureg.parse_expression(units)
        quantity = value * units

        return cls(name, quantity,
                   sources=sources,
                   references=references,
                   assumptions=assumptions)


# TODO: move this... still sketching out implementation
#loader = PropertyLoader.from_path()
#PROPERTIES = loader.properties
#PropertName = loader.property_name