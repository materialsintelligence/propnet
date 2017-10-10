from typing import *
from glob import glob
from enum import Enum
from os import path
from monty.serialization import loadfn
from propnet import logger

from propnet.core.properties import PropertyMetadata, parse_property

# auto loading of allowed properties

files: List[str] = glob(path.join(path.dirname(__file__), '../properties/*.yaml'))

property_metadata: List[Tuple[str, PropertyMetadata]] = []
for f in files:
    try:
        property_metadata.append(parse_property(loadfn(f)))
    except Exception as e:
        logger.error('Failed to parse {}, {}.'.format(path.basename(f), e))

# using property names for Enum, conventional to have all upper case
# but using all lower case here
PropertyType: Enum = Enum('PropertyType', property_metadata)

# for convenience
all_property_names: Tuple[str] = (p.name for p in PropertyType)