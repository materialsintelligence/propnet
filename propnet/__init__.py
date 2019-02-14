import logging
import sys

from pint import UnitRegistry
from io import StringIO

# module-wide logger
logger = logging.getLogger(__name__)
print_logger = logging.getLogger(__name__ + "_print_log")

# log to string, useful for web view
# TODO: just move this to the web view ...
log_stream = StringIO()
log_handler = logging.StreamHandler(stream=log_stream)
log_handler.setLevel(logging.WARNING)
logger.addHandler(log_handler)

# make sure we see our log messages in Jupyter too
jupyter_log_handler = logging.StreamHandler(stream=sys.stdout)
jupyter_log_handler.setLevel(logging.WARNING)
logger.addHandler(jupyter_log_handler)

# stream to capture output from rogue print statements
print_stream = StringIO()
print_handler = logging.StreamHandler(stream=print_stream)
print_handler.setLevel(logging.INFO)
print_logger.setLevel(logging.INFO)
print_logger.addHandler(print_handler)


logger.warning("Propnet is not intended for public use at this time. "
               "Functionality might change.\n")

# module-wide unit registry
ureg = UnitRegistry()
# add atoms as a unit-less quantity to our unit registry, e.g. for eV/atom
ureg.define('atom = []')
ureg.define('Rydberg = 13.605693009 * eV = Ry')  # from CODATA 13.605 693 009(84) eV
ureg.define('USD = [currency]')
