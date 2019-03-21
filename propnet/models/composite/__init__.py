from pkgutil import iter_modules
from propnet.core.models import PyModuleCompositeModel

# This list is to test if we have models with the same name
_COMPOSITE_MODEL_NAMES_LIST = []


def add_builtin_models_to_registry(register_symbols=True):
    _COMPOSITE_MODEL_NAMES_LIST.clear()
    # Load composite models
    composite_module_list = iter_modules(__path__)
    if register_symbols:
        from propnet.symbols import add_builtin_symbols_to_registry
        add_builtin_symbols_to_registry()
    for _, module_name, _ in composite_module_list:
        module_path = "propnet.models.composite.{}".format(module_name)
        model = PyModuleCompositeModel(module_path, is_builtin=True, overwrite_registry=True)
        globals()[model.name] = model
        _COMPOSITE_MODEL_NAMES_LIST.append(model.name)


add_builtin_models_to_registry()
