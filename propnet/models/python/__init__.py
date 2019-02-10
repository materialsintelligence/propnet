from pkgutil import iter_modules
from propnet.core.models import PyModuleModel
from propnet.core.registry import Registry


DEFAULT_MODEL_DICT = Registry("models")

# Load python models
MODULE_LIST = iter_modules(__path__)
for _, module_name, _ in MODULE_LIST:
    module_path = "propnet.models.python.{}".format(module_name)
    model = PyModuleModel(module_path)
    DEFAULT_MODEL_DICT.update({model.name: model})
