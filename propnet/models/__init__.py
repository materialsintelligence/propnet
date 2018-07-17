from pkgutil import iter_modules
import os
from propnet.core.models import EquationModel, PyModuleModel
from propnet.models import python


DEFAULT_MODELS = []

# Load equation models
equation_model_dir = os.path.join(os.path.dirname(__file__), "serialized")
equation_model_files = os.listdir(equation_model_dir)
for filename in equation_model_files:
    model = EquationModel.from_file(filename)
    DEFAULT_MODELS.append(model)

# Load python models
modules = iter_modules(python.__file__)
for _, module_name, _ in modules:
    module_path = "propnet.models.python.{}".format(module_name)
    DEFAULT_MODELS.append(PyModuleModel(module_path))
