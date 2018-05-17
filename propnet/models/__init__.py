from os.path import dirname, basename, isfile
from glob import glob

_DEFAULT_MODEL_FILES = glob(dirname(__file__) + "/*.py")

DEFAULT_MODEL_NAMES = [basename(f)[:-3] for f in _DEFAULT_MODEL_FILES
                       if isfile(f) and not basename(f).startswith('_')]

# auto loading of defined models
# it's a bit hack-y due to wanting to store each class in a separate file
# but also have them available in the propnet.models namespace
DEFAULT_MODELS = {}
for model in DEFAULT_MODEL_NAMES:
    model_cls = getattr(__import__('propnet.models.{}'.format(model),
                                   fromlist=['model']), model)
    locals()[model] = model_cls
    DEFAULT_MODELS[model] = model_cls