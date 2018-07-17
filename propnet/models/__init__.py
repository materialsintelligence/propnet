from pkgutil import iter_modules
import os
from propnet.core.models import EquationModel, PyModuleModel
from propnet.models import python


DEFAULT_MODELS = []

# Load equation models
equation_model_dir = os.path.join(os.path.dirname(__file__), "serialized")
equation_model_files = os.listdir(equation_model_dir)
for filename in equation_model_files:
    model_path = os.path.join(equation_model_dir, filename)
    model = EquationModel.from_file(model_path)
    DEFAULT_MODELS.append(model)

# Load python models
modules = iter_modules(python.__path__)
for _, module_name, _ in modules:
    module_path = "propnet.models.python.{}".format(module_name)
    DEFAULT_MODELS.append(PyModuleModel(module_path))

DEFAULT_MODEL_DICT = {d.name: d for d in DEFAULT_MODELS}
DEFAULT_MODEL_NAMES = list(DEFAULT_MODEL_DICT.keys())