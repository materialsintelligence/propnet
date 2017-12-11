from typing import *
from glob import glob
from enum import Enum
from os import path
from monty.serialization import loadfn
from propnet import logger

from propnet.core.properties import PropertyMetadata

# TODO: clean up this file, move as much to propnet.core.properties as possible

# Auto loading of all allowed properties

# Stores all loaded properties as PropertyMetadata instances in a dictionary, mapped to
# their names
property_metadata: Dict[str, PropertyMetadata] = {}

property_metadata_files: List[str] = glob(path.join(path.dirname(__file__), '../properties/*.yaml'))

for f in property_metadata_files:
    try:
        metadata = PropertyMetadata.from_dict(loadfn(f))
        property_metadata[metadata.name] = metadata
    except Exception as e:
        logger.error('Failed to parse {}, {}.'.format(path.basename(f), e))

# using property names for Enum, conventional to have all upper case
# but using all lower case here
PropertyType: Enum = Enum('PropertyType', [(k, v) for k, v in property_metadata.items()])

# Stores all loaded properties' names in a tuple in the global scope.
all_property_names: Tuple[str] = tuple(p for p in property_metadata.keys())

def get_display_name(property_name):
    """Convenience function """
    return PropertyType[property_name].value.display_names[0]