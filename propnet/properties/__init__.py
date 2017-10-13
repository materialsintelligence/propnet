from typing import *
from glob import glob
from enum import Enum
from os import path
from monty.serialization import loadfn
from propnet import logger

from propnet.core.properties import PropertyMetadata

# Auto loading of all allowed properties

# Stores all loaded properties as PropertyMetadata instances in a dictionary in the global scope, mapped from
# their names.
property_metadata: dict[str : PropertyMetadata] = {}

property_metadata_files: List[str] = glob(path.join(path.dirname(__file__), '../properties/*.yaml'))

for f in property_metadata_files:
    try:
        storing = PropertyMetadata.from_dict(loadfn(f))
        property_metadata[storing.name] = storing
    except Exception as e:
        logger.error('Failed to parse {}, {}.'.format(path.basename(f), e))

# using property names for Enum, conventional to have all upper case
# but using all lower case here
PropertyType: Enum = Enum('PropertyType', property_metadata.keys())

# Stores all loaded properties' names in a tuple in the global scope.
all_property_names: Tuple[str] = tuple(p for p in property_metadata.keys())