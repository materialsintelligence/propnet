"""
Module containing classes and methods for Model functionality in Propnet code.
"""

import numpy as np
import os
from abc import ABC, abstractmethod

from monty.serialization import loadfn, dumpfn
from monty.json import MSONable

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from propnet.symbols import DEFAULT_SYMBOLS
from propnet import logger, ureg
from propnet.core.utils import uuid, references_to_bib
from propnet.core.exceptions import ModelEvaluationError, IncompleteData


# TODO: Constraints are really just models that output True/False
#       can we refactor with this?
class Model(ABC):
    """
    Abstract model class for all models appearing in Propnet

    Args:
        name (str): title of the model
        connections (dict): list of connections dictionaries,
            which take the form {"inputs": [Symbols], "outputs": [Symbols]},
            for example:
            connections = [{"inputs": ["p", "T"], "outputs": ["V"]},
                           {"inputs": ["T", "V"], "outputs": ["p"]}]
        constraints (str): title
        description (str): long form description of the model
        categories (str): list of categories applicable to
            the model
        references ([str]): list of the informational links
            explaining / supporting the model

    """
    def __init__(self, name, connections, constraints=None, description=None,
                 categories=None, references=None):
        self.name = name
        self.connections = connections
        self.description = description
        self.categories = categories
        self.references = references
        self.constraints = constraints

    @abstractmethod
    def plug_in(self, symbol_value_dict):
        """
        Plugs in a symbol to quantity dictionary

        Args:
            symbol_value_dict ({symbol: value}): a mapping
                of symbols to values to be substituted
                into the model to yield output

        Returns:
            dictionary of output symbols with associated
                values generated from the input
        """
        return

    @property
    def inputs(self):
        return [d['inputs'] for d in self.connections]

    @property
    def outputs(self):
        return [d['outputs'] for d in self.connections]


class EquationModel(Model, MSONable):
    """
    Equation model is a Model subclass which is invoked
    from a list of equations

    Args:
        name (str): title of the model
        connections (dict): list of connections dictionaries,
            which take the form {"inputs": [Symbols], "outputs": [Symbols]},
            for example:
            connections = [{"inputs": ["p", "T"], "outputs": ["V"]},
                           {"inputs": ["T", "V"], "outputs": ["p"]}]
        constraints (str): title
        description (str): long form description of the model
        categories (str): list of categories applicable to
            the model
        references ([str]): list of the informational links
            explaining / supporting the model

    """
    def __init__(self, name, equations, connections, symbol_map=None,
                 constraints=None, description=None, categories=None,
                 references=None):
        self.equations = equations
        self.symbol_map = symbol_map
        super(EquationModel, self).__init__(
            name, connections, constraints, description,
            categories, references)

    # TODO: shouldn't this respect/use connections info,
    #       or is that done elsewhere?
    def plug_in(self, symbol_value_dict):
        # Parse equations and substitute
        eqns = [parse_expr(eq) for eq in self.equations]
        eqns = [eqn.subs(symbol_value_dict) for eqn in eqns]
        possible_outputs = set()
        for eqn in eqns:
            possible_outputs = possible_outputs.union(eqn.free_symbols)
        outputs = {}
        # Determine outputs from solutions to substituted equations
        for possible_output in possible_outputs:
            solutions = sp.nonlinsolve(eqns, possible_output)
            # taking first solution only, and only asking for one output symbol
            # so know length of output tuple for solutions will be 1
            solution = list(solutions)[0][0]
            if not isinstance(solution, sp.EmptySet):
                outputs[str(possible_output)] = float(sp.N(solution))
        return outputs

    @classmethod
    def from_file(cls, filename):
        """Load model from file"""
        model = loadfn(filename)
        if isinstance(model, Model):
            return model
        else:
            return cls.from_dict(model)

    @classmethod
    def from_preset(cls, name):
        """Loads from preset library of models"""
        loc = os.path.join("..", "models", "{}.yaml".format(name))
        if os.path.isfile(loc):
            return cls.from_file(loc)
        else:
            raise ValueError("No {} model found at {}".format(name, loc))


class PyModel(Model):
    """
    Purely python based model which allows for a flexible "plug_in"
    method as input, then invokes that method in the defined plug-in
    method
    """
    def __init__(self, name, connections, plug_in, constraints=None,
                 description=None, categories=None, references=None):
        self._plug_in = plug_in
        super(PyModel, self).__init__(
            name, connections, constraints, description,
            categories, references)

    def plug_in(self, symbol_value_dict):
        return self._plug_in(symbol_value_dict)


# Note that this class exists purely as a factory method for PyModel
# which could be implemented as a class method of PyModel
# but wouldn't serialize as cleanly
class PyModuleModel(PyModel):
    def __init__(self, module_path):
        self._module_path = module_path
        mod = __import__(module_path, globals(), locals(), ['config'], 0)
        super(PyModuleModel, self).__init__(**mod.config)

    def as_dict(self):
        return {"module_path": self._module_path,
                "@module": "propnet.core.model",
                "@class": "PyModuleModel"}