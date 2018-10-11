"""
Module defining exception and warning classes.
"""


class ModelEvaluationError(RuntimeError):
    """The model failed to evaluate."""
    pass


class IncompleteData(Warning):
    """When necessary data is missing for a given model or symbol."""
    pass


class SymbolConstraintError(RuntimeError):
    """Invalid quantity value with respect to symbol constraints"""
    pass