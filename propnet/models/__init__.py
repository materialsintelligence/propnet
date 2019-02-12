from propnet.models import serialized, python, composite
from propnet.core.registry import Registry

# This is just to enable importing the model directly from this module for example code generation
for name, model in Registry("models").items():
    globals()[name] = model
