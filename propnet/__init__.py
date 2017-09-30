import logging
from pint import UnitRegistry
from typing import Union, Sequence, SupportsFloat, Tuple

logger = logging.getLogger(__name__)
ureg = UnitRegistry()

Tensor = Union[SupportsFloat, Sequence[SupportsFloat], Sequence[Sequence[SupportsFloat]]]
QuantityLike = Union[ureg.Quantity, str, Tuple[Tensor, str]]