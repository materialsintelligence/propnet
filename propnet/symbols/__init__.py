import os

from glob import glob
from monty.serialization import loadfn

from propnet.core.symbols import Symbol

# Auto loading of all allowed properties

# stores all loaded properties as PropertyMetadata instances in a dictionary,
# mapped to their names
DEFAULT_SYMBOLS = {}
DEFAULT_SYMBOL_VALUES = {}

_DEFAULT_SYMBOL_TYPE_FILES = glob(
    os.path.join(os.path.dirname(__file__), '../symbols/**/*.yaml'),
    recursive=True)

for f in _DEFAULT_SYMBOL_TYPE_FILES:
    symbol_type = Symbol.from_dict(loadfn(f))
    DEFAULT_SYMBOLS[symbol_type.name] = symbol_type
    if symbol_type.default_value is not None:
        DEFAULT_SYMBOL_VALUES[symbol_type] = symbol_type.default_value
    if "{}.yaml".format(symbol_type.name) not in f:
        raise ValueError('Name/filename mismatch in {}'.format(f))

# Stores all loaded properties' names in a tuple in the global scope.
DEFAULT_UNITS = {name: symbol.units.format_babel() if symbol.units else None
                 for name, symbol in DEFAULT_SYMBOLS.items()}
DEFAULT_SYMBOL_TYPE_NAMES = tuple(DEFAULT_SYMBOLS.keys())

for name, symbol in DEFAULT_SYMBOLS.items():
    globals()[name] = symbol