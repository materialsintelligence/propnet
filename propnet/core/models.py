"""
Module containing classes and methods for Model functionality in Propnet code.
"""

import numpy as np
import os

from atomate.utils.utils import load_class
from monty.serialization import loadfn, dumpfn
from monty.json import MSONable

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from propnet.symbols import DEFAULT_SYMBOLS
from propnet import logger, ureg
from propnet.core.utils import uuid, references_to_bib
from propnet.core.exceptions import ModelEvaluationError, IncompleteData


class Model(MSONable):
    def __init__(self, name, symbols, function, constraints=None,
                 description=None, categories=None, references=None):
        self.symbols = symbols
        self.name = name
        self.description = description
        self.categories = categories
        self.references = references
        self.constraints = constraints
        # If function is a string, process into input function
        if isinstance(function, str):
            self._function_string = function
            self.function = sp.lambdify(function)
        else:
            self.function = function
            self._function_string = None

    def run(self, *args, **kwargs):
        """Runs the function associated with the model"""
        return self.function(*args, **kwargs)

    def validate(self, *args, **kwargs):
        """Validates constraints"""
        for constraint in self.constraints:
            if not constraint.validate(*args, **kwargs):
                return False
        return True


# TODO: Might consider these as purely factory methods
#       this implementation saves us some space on serialization
def PyModel(Model):
    def __init__(self, module_path):
        self._module_path = module_path
        mod = __import__(module_path, globals(), locals(), ['config'], 0)
        return super(PyModel, self).__init__(**mod.config)

def MSONModel(Model):
    def __init__(self, filename):
        pass

    def to_file(self, filename):
        """Dumps model to file"""
        if self._function_string is None:
            raise ValueError("Model can only be serialized if input function "
                             "is a string")
        d = self.as_dict()
        d['function'] = self._function_string
        dumpfn(d, filename)

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
            try:
                return load_class("models.pymodels.{}".format(name))
            except ImportError:
                raise ValueError("No {} model found models or "
                                 "models.pymodels".format(name))
