# noinspection PyUnresolvedReferences
import propnet.symbols
from propnet.models import serialized, python, composite
from propnet.core.registry import Registry


# This is just to enable importing the model directly from this module for example code generation
def _update_globals():
    for name, model in Registry("models").items():
        if model.is_builtin:
            globals()[name] = model


def add_builtin_models_to_registry(register_symbols=True):
    if register_symbols:
        propnet.symbols.add_builtin_symbols_to_registry()
    serialized.add_builtin_models_to_registry(register_symbols=False)
    python.add_builtin_models_to_registry(register_symbols=False)
    composite.add_builtin_models_to_registry(register_symbols=False)
    _update_globals()


_update_globals()
