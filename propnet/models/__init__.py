from os.path import dirname, basename, isfile
from glob import glob

files = glob(dirname(__file__)+"/*.py")

# TODO: replace with dict of name to class ?
all_model_names = [basename(f)[:-3] for f in files
                   if isfile(f) and not basename(f).startswith('_')]

# auto loading of defined models
# it's a bit hack-y due to wanting to store each class in a separate file
# but also have them available in the propnet.models namespace
for model in all_model_names:
    model_cls = getattr(__import__('propnet.models.{}'.format(model), fromlist=['model']), model)
    locals()[model] = model_cls