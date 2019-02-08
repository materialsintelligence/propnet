"""
Module containing classes and methods for provenance generation and parsing
"""

from monty.json import MSONable


class ProvenanceElement(MSONable):
    """
    Tree-like data structure for representing provenance.
    """

    __slots__ = ['_model', '_inputs', '_source']

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
        if isinstance(model, str):
            self._model = model
        else:
            self._model = getattr(model, 'name', model)

        if inputs is not None:
            if not isinstance(inputs, list):
                try:
                    inputs = [x for x in inputs]
                except TypeError:
                    inputs = [inputs]

        self._inputs = inputs
        self._source = source

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, rhv):
        self._model = rhv

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, rhv):
        self._inputs = rhv

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, rhv):
        self._source = rhv

    def __str__(self):
        pre = ",".join([
            "<{}, {}, {}>".format(q._symbol_type.name, q.value, q._provenance)
            for q in self.inputs])
        return "{{{}: [{}]}}".format(self.model, pre)

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        # Ignoring metadata in source. May need to revisit?
        return self.model == other.model and \
               set(self.inputs or []) == set(other.inputs or [])

    def __hash__(self):
        hash_value = hash(self.model or 0)
        for v in self.inputs or []:
            hash_value = hash_value ^ hash(v)
        return hash_value


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
