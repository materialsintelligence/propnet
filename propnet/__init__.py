import logging
import sys

from pint import UnitRegistry
from io import StringIO

# module-wide logger
logger = logging.getLogger(__name__)

# log to string, useful for web view
# TODO: just move this to the web view ...
log_stream = StringIO()
logger.addHandler(logging.StreamHandler(stream=log_stream))

# make sure we see our log messages in Jupyter too
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

logger.warning("Propnet is not intended for public use at this time. "
               "Functionality might change.\n")

# module-wide unit registry
ureg = UnitRegistry()

# add atoms as a unit-less quantity to our unit registry, e.g. for eV/atom
ureg.define('atom = []')
ureg.define('Rydberg = 13.605693009 * eV = Ry')  # from CODATA 13.605 693 009(84) eV