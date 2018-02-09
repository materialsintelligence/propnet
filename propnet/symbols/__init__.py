import os

from typing import *
from glob import glob
from enum import Enum
from monty.serialization import loadfn
from propnet import logger

from propnet.core.symbols import SymbolMetadata

# TODO: clean up this file, move as much to propnet.core.properties as possible

# Auto loading of all allowed properties

# Stores all loaded properties as PropertyMetadata instances in a dictionary, mapped to
# their names
_symbol_metadata: Dict[str, SymbolMetadata] = {}

_symbol_metadata_files: List[str] = glob(os.path.join(os.path.dirname(__file__),
                                                    '../symbols/**/*.yaml'),
                                         recursive=True)

for f in _symbol_metadata_files:
    try:
        metadata = SymbolMetadata.from_dict(loadfn(f))
        _symbol_metadata[metadata.name] = metadata
        if "{}.yaml".format(metadata.name) not in f:
            raise ValueError('Name/filename mismatch in {}'.format(f))
    except Exception as e:
        logger.error('Failed to parse {}, {}.'.format(path.basename(f), e))

# using property names for Enum, conventional to have all upper case
# but using all lower case here
SymbolType: Enum = Enum('SymbolType', [(k, v) for k, v in _symbol_metadata.items()])

# Stores all loaded properties' names in a tuple in the global scope.
all_symbol_names: Tuple[str] = tuple(p for p in _symbol_metadata.keys())

def get_display_name(symbol_name):
    """Convenience function

    Args:
      symbol_name: 

    Returns:

    """
    return SymbolType[symbol_name].value.display_names[0]