"""
Module containing classes and methods for provenance generation and parsing
"""

from monty.json import MSONable


class ProvenanceElement(MSONable):
    """
    Tree-like data structure for representing provenance.
    """

    __slots__ = ['m', 'inputs']

    def __init__(self, model=None, inputs=None, source=None):
        """
        Args:
            model: (Model) model that outputs the quantity object this
                ProvenanceElement is attached to.
            inputs: (list<Quantity>) quantities fed in to the model
                to generate the quantity object this ProvenanceElement
                is attached to.
            source: static source, e. g. Materials Project
        """
        self.model = getattr(model, 'name', model)
        self.inputs = inputs

    def __str__(self):
        pre = ",".join([
            "<{}, {}, {}>".format(q._symbol_type.name, q.value, q._provenance)
            for q in self.inputs])
        return "{{{}: [{}]}}".format(self.model, pre)


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
            symbol: (Symbol) we are searching for paths from
                this symbol to head.
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