import numpy as np

from typing import List, Dict, SupportsFloat, NamedTuple, Optional, Union
from propnet import logger, ureg
from glob import glob
from enum import Enum
from monty.serialization import loadfn
from pybtex.database.input.bibtex import Parser

Property = NamedTuple('Property', [('name', str),
                                   ('unit', ureg.Quantity),
                                   ('display_names', List[str]),
                                   ('display_symbols', List[str]),
                                   ('dimension', List),
                                   ('test_value', np.ndarray),
                                   ('comment', str)])

PropertyName: Optional[Enum] = None
PROPERTIES: Optional[Dict[PropertyName, Property]] = None

class PropertyLoader():

    def __init__(self, properties: List[Dict]):
        """

        :param properties:
        """

        self.properties = [self.parse_property(property)
                           for property in properties]

        self.property_name = Enum('PropertyName',
                                  [property.name for property in self.properties])

    @classmethod
    def from_path(cls, path: str):
        """

        :param path:
        :return:
        """

        files = glob("{}/*.yaml".format(path))
        properties = [loadfn(f) for f in files]

        return cls(properties)

    def parse_property(self, property: dict) -> Property:
        """

        :param property:
        :return:
        """

        property = Property(**property)

        # check canonical name
        if not property.name.isidentifier() or \
           not property.name.islower():
            raise ValueError("The canonical name ({}) is not valid."
                             .format(property.name))

        # check units supported by pint
        units = ureg.Quantity.from_tuple((1, )+tuple(tuple(x) for x in property.unit))

        # check required fields
        if len(property.display_names) == 0:
            raise ValueError("Please provide at least one display name for {}."
                             .format(property.name))
        if len(property.display_symbols) == 0:
            raise ValueError("Please provide at least one display symbol for {}."
                             .format(property.name))

        # check test value is valid
        test_value = float(property.test_value)
        if test_value == 0:
            logger.warn("Test value for {} is 0, is there a more appropriate test value?"
                        .format(property.name))

        # check dimensions are valid
        try:
            empty_array = np.zeros(property.dimension)
        except:
            raise ValueError("Invalid dimensions for {}.".format(property.name))

        return Property


class PropertyInstance:

    def __init__(self,
                 name: Union[str, PropertyName],
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
                         name: PropertyName,
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
loader = PropertyLoader.from_path()
PROPERTIES = loader.properties
PropertName = loader.property_name