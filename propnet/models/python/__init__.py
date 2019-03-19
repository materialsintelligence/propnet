from pkgutil import iter_modules
from propnet.core.models import PyModuleModel


def add_builtin_models_to_registry(readd_symbols=True):
    # Load python models
    MODULE_LIST = iter_modules(__path__)
    if readd_symbols:
        from propnet.symbols import add_builtin_symbols_to_registry
        add_builtin_symbols_to_registry()
    for _, module_name, _ in MODULE_LIST:
        module_path = "propnet.models.python.{}".format(module_name)
        PyModuleModel(module_path, is_builtin=True, overwrite_registry=True)


add_builtin_models_to_registry()
