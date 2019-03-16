from pkgutil import iter_modules
from propnet.core.models import PyModuleCompositeModel
from propnet.core.registry import Registry
# from propnet.models import python

COMPOSITE_MODEL_DICT = Registry("composite_models")

# Load composite models
COMPOSITE_MODULE_LIST = iter_modules(__path__)


def add_builtin_models_to_registry():
    for _, module_name, _ in COMPOSITE_MODULE_LIST:
        module_path = "propnet.models.composite.{}".format(module_name)
        PyModuleCompositeModel(module_path, is_builtin=True)


add_builtin_models_to_registry()
