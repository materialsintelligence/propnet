import os
from propnet.core.models import EquationModel
from propnet.core.registry import Registry
from glob import glob

DEFAULT_MODEL_DICT = Registry("models")

# Load equation models
EQUATION_MODEL_DIR = os.path.join(os.path.dirname(__file__))
EQUATION_MODULE_FILES = glob(EQUATION_MODEL_DIR + '/*.yaml')


def add_builtin_models_to_registry():
    # noinspection PyUnresolvedReferences
    import propnet.symbols
    for filename in EQUATION_MODULE_FILES:
        model_path = os.path.join(EQUATION_MODEL_DIR, filename)
        EquationModel.from_file(model_path, is_builtin=True)


add_builtin_models_to_registry()
