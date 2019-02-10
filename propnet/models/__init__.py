from propnet.models import serialized, python, composite
from propnet.core.registry import Registry

DEFAULT_MODEL_DICT = Registry("models")
COMPOSITE_MODEL_DICT = Registry("composite_models")

COMPOSITE_MODEL_NAMES = list(COMPOSITE_MODEL_DICT.keys())
DEFAULT_MODEL_NAMES = list(DEFAULT_MODEL_DICT.keys())

DEFAULT_MODELS = list(DEFAULT_MODEL_DICT.values())
DEFAULT_COMPOSITE_MODELS = list(COMPOSITE_MODEL_DICT.values())

# This is just to enable importing the model directly from this module for example code generation
for name, model in DEFAULT_MODEL_DICT.items():
    globals()[name] = model
