import os
from propnet.core.models import EquationModel
from glob import glob

# This list is to test if we have models with the same name
_EQUATION_MODEL_NAMES_LIST = []


def add_builtin_models_to_registry(register_symbols=True):
    _EQUATION_MODEL_NAMES_LIST.clear()
    # Load equation models
    equation_model_dir = os.path.join(os.path.dirname(__file__))
    equation_module_files = glob(equation_model_dir + '/*.yaml')

    if register_symbols:
        from propnet.symbols import add_builtin_symbols_to_registry
        add_builtin_symbols_to_registry()
    for filename in equation_module_files:
        model_path = os.path.join(equation_model_dir, filename)
        model = EquationModel.from_file(model_path, is_builtin=True, overwrite_registry=True)
        globals()[model.name] = model
        _EQUATION_MODEL_NAMES_LIST.append(model.name)


add_builtin_models_to_registry()
