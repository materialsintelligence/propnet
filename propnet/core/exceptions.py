"""
Module defining exception and warning classes.
"""


class ModelEvaluationError(RuntimeError):
    """The model failed to evaluate."""
    pass


class IncompleteData(Exception):
    """When necessary data is missing for a given model or symbol."""
    pass