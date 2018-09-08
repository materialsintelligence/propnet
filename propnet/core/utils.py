import numpy as np
from monty.json import MSONable
from uncertainties import unumpy

from propnet.core.quantity import Quantity
from propnet.core.models import Model


def weighted_mean(quantities):
    """
    Function to retrieve weighted mean

    Args:
        quantities ([Quantity]): list of quantities

    Returns:
        weighted mean
    """
    # can't run this twice yet ...
    # TODO: remove
    if hasattr(quantities[0].value, "std_dev"):
        return quantities

    input_symbol = quantities[0].symbol
    if input_symbol.category == 'object':
        # TODO: can't average 'objects', highlights a weakness in Quantity class
        # would be fixed by changing this class design ...
        return quantities

    if not all(input_symbol == q.symbol for q in quantities):
        raise ValueError("Can only calculate a weighted mean if all quantities "
                         "refer to the same symbol.")

    # TODO: an actual weighted mean; just a simple mean at present
    # TODO: support propagation of uncertainties (this will only work once at present)

    # TODO: test this with units, not magnitudes ... remember units may not be canonical units(?)
    if isinstance(quantities[0].value, list):
        # hack to get arrays working for now
        vals = [q.value for q in quantities]
    else:
        vals = [q.value.magnitude for q in quantities]

    new_magnitude = np.mean(vals, axis=0)
    std_dev = np.std(vals, axis=0)
    new_value = unumpy.uarray(new_magnitude, std_dev)

    new_tags = set()
    new_provenance = ProvenanceElement(model='aggregation', inputs=[])
    for q in quantities:
        if q.tags:
            for t in q.tags:
                new_tags.add(t)
        new_provenance.inputs.append(q)

    new_quantity = Quantity(symbol_type=input_symbol,
                            value=new_value,
                            tags=list(new_tags),
                            provenance=new_provenance)

    return new_quantity


class ProvenanceElement(MSONable):
    """
    Tree-like data strucutre for representing provenance.
    """

    __slots__ = ['m', 'inputs']

    def __init__(self, model=None, inputs=None):
        """
        Args:
            model: (Model) model that outputs the quantity object this
                       ProvenanceElement is attached to.
            inputs: (list<Quantity>) quantities fed in to the model
                                     to generate the quantity object
                                     this ProvenanceElement is attached to.
        """
        self.model = model.name if isinstance(model, Model) else model
        self.inputs = inputs

    def __str__(self):
        x = ""
        for q in self.inputs:
            x += "<" + q._symbol_type.name + ", " + str(q.value) + ", " + str(q._provenance) + ">,"
        return "{" + self.m.name + ": [" + x + "]}"


class SymbolTree(object):
    """
    Wrapper around TreeElement data structure for export from
    the method, encapsulating functionality.
    """

    __slots__ = ['head']

    def __init__(self, head):
        """
        Args:
            head: (TreeElement) head of the tree.
        """
        self.head = head

    def get_paths_from(self, symbol):
        """
        Gets all paths from input to symbol
        Args:
            symbol: (Symbol) we are searching for paths from this symbol to head.
        Returns:
            (list<SymbolPath>)
        """
        to_return = []
        visitation_queue = [self.head]
        while len(visitation_queue) != 0:
            visiting = visitation_queue.pop(0)
            for elem in visiting.children:
                visitation_queue.append(elem)
            if symbol in visiting.inputs:
                v = visiting
                model_trail = []
                while v.parent is not None:
                    model_trail.append(v.m)
                    v = v.parent
                to_return.append(SymbolPath(visiting.inputs, model_trail))
        return to_return


class TreeElement(object):
    """
    Tree-like data structure for representing property
    relationship paths.
    """

    __slots__ = ['m', 'inputs', 'parent', 'children']

    def __init__(self, m, inputs, parent, children):
        """
        Args:
            m: (Model) model outputting the parent from children inputs
            inputs: (set<Symbol>) Symbol inputs required to produce the parent
            parent: (TreeElement)
            children: (list<TreeElement>) all PathElements derivable from this one
        """
        self.m = m
        self.inputs = inputs
        self.parent = parent
        self.children = children


class SymbolPath(object):
    """
    Utility class to store elements of a Symbol path through
    various inputs and outputs.
    """

    __slots__ = ['symbol_set', 'model_path']

    def __init__(self, symbol_set, model_path):
        """
        Args:
            symbol_set: (set<Symbol>) set of all inputs required to complete the path
            model_path: (list<Model>) list of models, in order, required to complete the path
        """
        self.symbol_set = symbol_set
        self.model_path = model_path

    def __eq__(self, other):
        if not isinstance(other, SymbolPath):
            return False
        if not self.symbol_set == other.symbol_set:
            return False
        if not self.model_path == other.model_path:
            return False
        return True