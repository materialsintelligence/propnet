from enum import Enum
from collections import namedtuple

ALLOWED_NODE_TYPES = ('Material', 'Symbol', 'Quantity', 'Model')


# TODO: this should be replaced with a proper class
NodeType = Enum('NodeType', ALLOWED_NODE_TYPES)
Node = namedtuple('Node', ['node_type', 'node_value'])
Node.__repr__ = lambda self: "{}<{}>".format(self.node_type.name,
                                             self.node_value.__repr__())