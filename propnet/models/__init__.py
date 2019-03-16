# noinspection PyUnresolvedReferences
import propnet.symbols
from propnet.models import serialized, python, composite
from propnet.core.registry import Registry

# This is just to enable importing the model directly from this module for example code generation
for name, model in Registry("models").items():
    globals()[name] = model


def add_builtin_models_to_registry():
    serialized.add_builtin_models_to_registry()
    python.add_builtin_models_to_registry()
    composite.add_builtin_models_to_registry()
