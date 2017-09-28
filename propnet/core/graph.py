import networkx as nx

from typing import NamedTuple, Any
from enum import Enum

NodeType = Enum('NodeType', ['Material',
                             'PropertyType',
                             'PropertyInstance',
                             'PymatgenObject',
                             'String',
                             'Model'])

Node = NamedTuple('Node', [('type', NodeType)
                           ('value', Any)])