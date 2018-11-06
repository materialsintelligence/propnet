from pkgutil import iter_modules
import os
from propnet.core.models import EquationModel, PyModuleModel,\
    PyModuleCompositeModel
from propnet.models import python, composite
from glob import glob


DEFAULT_MODELS = []
DEFAULT_COMPOSITE_MODELS = []

# Load equation models
EQUATION_MODEL_DIR = os.path.join(os.path.dirname(__file__), "serialized")
EQUATION_MODULE_FILES = glob(EQUATION_MODEL_DIR+'/*.yaml')
for filename in EQUATION_MODULE_FILES:
    model_path = os.path.join(EQUATION_MODEL_DIR, filename)
    model = EquationModel.from_file(model_path)
    DEFAULT_MODELS.append(model)

# Load python models
MODULE_LIST = iter_modules(python.__path__)
for _, module_name, _ in MODULE_LIST:
    module_path = "propnet.models.python.{}".format(module_name)
    DEFAULT_MODELS.append(PyModuleModel(module_path))

DEFAULT_MODEL_DICT = {d.name: d for d in DEFAULT_MODELS}
DEFAULT_MODEL_NAMES = list(DEFAULT_MODEL_DICT.keys())

# Load composite models
COMPOSITE_MODULE_LIST = iter_modules(composite.__path__)
for _, module_name, _ in COMPOSITE_MODULE_LIST:
    module_path = "propnet.models.composite.{}".format(module_name)
    DEFAULT_COMPOSITE_MODELS.append(PyModuleCompositeModel(module_path))

COMPOSITE_MODEL_DICT = {c.name: c for c in DEFAULT_COMPOSITE_MODELS}
COMPOSITE_MODEL_NAMES = list(COMPOSITE_MODEL_DICT.keys())

# Convenience function for loading a specific model by name
def load_default_model(name):
    """Helper method to load a default model from a name"""
    return DEFAULT_MODEL_DICT[name]
