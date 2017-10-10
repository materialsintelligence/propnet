import logging
from pint import UnitRegistry
from io import StringIO

log_stream = StringIO()

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(stream=log_stream))
ureg = UnitRegistry()