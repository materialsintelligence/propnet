import numpy as np

from typing import List, NamedTuple, Optional
from propnet import logger, ureg
from pybtex.database.input.bibtex import Parser

# metadata associated with each PropertyType
PropertyMetadata = NamedTuple('PropertyMetadata', [('units', ureg.Quantity),
                                                   ('display_names', List[str]),
                                                   ('display_symbols', List[str]),
                                                   ('dimension', List),
                                                   ('test_value', np.ndarray),
                                                   ('comment', str)])

def parse_property(property: dict) -> (str, PropertyMetadata):
    """

    :param property:
    :return:
    """

    name = property['name']
    # using the name as a key, no need to keep it in metadata too
    del property['name']

    # check canonical name
    if not name.isidentifier() or \
            not name.islower():
        raise ValueError("The canonical name ({}) is not valid."
                         .format(name))

    # check units supported by pint
    units = ureg.Quantity.from_tuple(property['units'])
    property['units'] = units

    # check required fields
    if len(property['display_names']) == 0:
        raise ValueError("Please provide at least one display name for {}."
                         .format(name))
    if len(property['display_symbols']) == 0:
        raise ValueError("Please provide at least one display symbol for {}."
                         .format(name))

    # check test value is valid
    test_value = np.array(property['test_value'])
    if test_value == 0:
        logger.warn("Test value for {} is 0, please change to a more appropriate test value."
                    .format(name))

    # check dimensions are valid
    try:
        empty_array = np.zeros(property['dimension'])
    except:
        raise ValueError("Invalid dimensions for {}.".format(name))

    return (name, PropertyMetadata(**property))

class Property:

    def __init__(self,
                 name: str,
                 quantity: ureg.Quantity,
                 sources: Optional[List] = None,
                 references: Optional[List[str]] = None):

        self.name = name
        self.quantity = quantity
        self.sources = sources

        parser = Parser()
        self.references = [parser.parse_string(s) for s in references]

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