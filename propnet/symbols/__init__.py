import os

from glob import glob
from monty.serialization import loadfn
from propnet import logger

from propnet.core.symbols import Symbol

# Auto loading of all allowed properties

# stores all loaded properties as PropertyMetadata instances in a dictionary,
# mapped to their names
DEFAULT_SYMBOLS = {}

_DEFAULT_SYMBOL_TYPE_FILES = glob(os.path.join(os.path.dirname(__file__),
                                           '../symbols/**/*.yaml'),
                                  recursive=True)

for f in _DEFAULT_SYMBOL_TYPE_FILES:
    try:
        symbol_type = Symbol.from_dict(loadfn(f))
        DEFAULT_SYMBOLS[symbol_type.name] = symbol_type
        if "{}.yaml".format(symbol_type.name) not in f:
            raise ValueError('Name/filename mismatch in {}'.format(f))
    except Exception as e:
        logger.error('Failed to parse {}, {}.'.format(os.path.basename(f), e))

# Stores all loaded properties' names in a tuple in the global scope.
DEFAULT_UNITS = {name: symbol.units
                 for name, symbol in DEFAULT_SYMBOLS.items()}
DEFAULT_SYMBOL_TYPE_NAMES = tuple(DEFAULT_SYMBOLS.keys())