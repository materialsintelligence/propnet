import os

from glob import glob
from monty.serialization import loadfn

from propnet.core.symbols import Symbol
from propnet.core.registry import Registry
# Auto loading of all allowed properties

# stores all loaded properties as PropertyMetadata instances in a dictionary,
# mapped to their names

_DEFAULT_SYMBOL_TYPE_FILES = glob(
    os.path.join(os.path.dirname(__file__), '../symbols/**/*.yaml'),
    recursive=True)

DEFAULT_SYMBOL_TYPE_NAMES = None


def _update_registry():
    for f in _DEFAULT_SYMBOL_TYPE_FILES:
        d = loadfn(f)
        d['is_builtin'] = True
        symbol_type = Symbol.from_dict(d)
        Registry("symbols")[symbol_type.name] = symbol_type
        if symbol_type.default_value is not None:
            Registry("symbol_values")[symbol_type] = symbol_type.default_value
        if "{}.yaml".format(symbol_type.name) not in f:
            raise ValueError('Name/filename mismatch in {}'.format(f))

    # Stores all loaded properties' names in a tuple in the global scope.
    Registry("units").update(
        {name: symbol.units.format_babel() if symbol.units else None
         for name, symbol in Registry("symbols").items()})


def _update_symbol_type_names():
    # TODO: Can we remove this?
    globals()['DEFAULT_SYMBOL_TYPE_NAMES'] = tuple(Registry("symbols").keys())


def add_builtin_symbols_to_registry():
    _update_registry()
    _update_symbol_type_names()


add_builtin_symbols_to_registry()

# This is just to enable importing this module
for name, symbol in Registry("symbols").items():
    globals()[name] = symbol
