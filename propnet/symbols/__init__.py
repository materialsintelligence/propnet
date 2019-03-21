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


def add_builtin_symbols_to_registry():
    for f in _DEFAULT_SYMBOL_TYPE_FILES:
        d = loadfn(f)
        d['is_builtin'] = True
        d['overwrite_registry'] = True
        symbol_type = Symbol.from_dict(d)
        if "{}.yaml".format(symbol_type.name) not in f:
            raise ValueError('Name/filename mismatch in {}'.format(f))

    # This is just to enable importing this module
    for name, symbol in Registry("symbols").items():
        if symbol.is_builtin:
            globals()[name] = symbol


add_builtin_symbols_to_registry()
