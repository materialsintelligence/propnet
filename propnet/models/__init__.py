from os.path import dirname, basename, isfile
from glob import glob

# auto loading of defined models

files = glob(dirname(__file__)+"/*.py")

all_model_names = tuple(basename(f)[:-3] for f in files
                        if isfile(f) and not basename(f).startswith('_'))

__all__ = all_model_names