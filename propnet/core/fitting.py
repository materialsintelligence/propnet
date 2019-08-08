"""Benchmarking routine for improved aggregation of properties.

This module contains routines to benchmark models against experimental values to improve the aggregation scheme
for quantities. By default, when a ``Material`` contains multiple derived quantities for a given property (symbol),
they are aggregated using a simple, unweighted mean. However, depending on the quality of the models used to produce
those quantities, this may not be ideal.

These routines calculate optimal weights for models given an experimental dataset of materials to match.

Example:
    >>> from propnet.core.fitting import fit_model_scores
    >>> from propnet.core.materials import Material
    >>> materials = [Material(...), ...]    # a list of materials populated with properties
    >>> benchmarks = [
    >>>     {'symbol_name': ...}, ... # a list of benchmark data as dicts
    >>> ]
    >>> # select models for which to calculate weights and run
    >>> scores = fit_model_scores(materials, benchmarks, models=['model_1', 'model_2', ...])
"""


# TODO: this could probably be renamed/reorganized
# TODO: this is very preliminary, could be improved substantially
from collections import OrderedDict

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

from propnet.core.quantity import QuantityFactory
# noinspection PyUnresolvedReferences
import propnet.models
from propnet.core.registry import Registry


def aggregate_quantities(quantities, model_score_dict=None):
    """
    Simple method for aggregating a set of quantities.

    Args:
        quantities (`iterable` of `propnet.core.quantity.NumQuantity`): iterable of Quantity objects to aggregate
        model_score_dict (dict): dict of weights to apply to models, keyed
            by model name or Model object

    Returns:
        propnet.core.quantity.NumQuantity: resulting quantity from aggregation
    """
    symbol = next(iter(quantities)).symbol
    if not all([q.symbol == symbol for q in quantities]):
        raise ValueError("Quantities passed to aggregate must be same symbol")
    weights = [get_weight(q, model_score_dict) for q in quantities]
    result_value = sum(
        [w * q.value for w, q in zip(weights, quantities)]) / sum(weights)
    return QuantityFactory.create_quantity(symbol, result_value)


def get_weight(quantity, model_score_dict=None):
    """
    Calculates weight based on scoring scheme and provenance of quantities.

    Args:
        quantity (propnet.core.quantity.NumQuantity): quantity for which to get weight
        model_score_dict (dict): dict of weights as floats to apply to models, keyed
            by model name or Model object

    Returns:
        float: calculated weight for input quantity
    """
    if quantity.provenance is None or quantity.provenance.inputs is None:
        return 1
    if model_score_dict is None:
        return 1
    weight = model_score_dict.get(quantity.provenance.model, 0)
    weight *= np.prod(
        [get_weight(q, model_score_dict) for q in quantity.provenance.inputs])
    return weight


# TODO: Add default graph when this is moved
def fit_model_scores(materials, benchmarks, models=None,
                     init_scores=None, constrain_sum=False):
    """
    Fits a set of model scores/weights to a set of benchmark data by minimizing the sum of squared errors
    with the benchmarking data.

    Args:
        materials (`list` of `propnet.core.materials.Material`): list of evaluated materials containing
            symbols for benchmarking
        benchmarks (`list` of `dict`): list of dicts, keyed by Symbol or symbol name, containing benchmark data
            for each material in ``materials``.
        models (`list` of `propnet.core.models.Model` or `list` of `str` or `None`): optional, list of models whose
            scores will be adjusted in the aggregation weighting scheme. Default: `None` (all models will be adjusted)
        init_scores (dict): optional, dict containing initial scores for minimization procedure, keyed by model name
            or Model. Scores are normalized to sum of scores. Default: `None` (all scores are equal)
        constrain_sum (bool): optional, ``True`` constrains the sum of scores to 1, ``False``
            removes this constraint. Default: ``False`` (no constraint)

    Returns:
        OrderedDict: dict of scores corresponding to the minimized sum of squared errors, keyed by model.
    """
    # Probably not smart to have ALL available models in the list. That's a lot of DOF.
    # TODO: Perhaps write a method to produce a list of models in the provenance trees
    #       of the symbols to be benchmarked. Should be easy with the caching we have for provenance.
    model_list = models or list(Registry("models").keys())

    def f(f_scores):
        model_score_dict = {m: s
                            for m, s in zip(model_list, f_scores)}
        return get_sse(materials, benchmarks, model_score_dict)

    scores = OrderedDict((m, 1) for m in model_list)
    scores.update(init_scores or {})
    x0 = np.array(list(scores.values()))
    x0 = x0 / np.sum(x0)
    bounds = Bounds([0]*len(x0), [1]*len(x0))
    if constrain_sum:
        constraint = [LinearConstraint([1]*len(x0), [1], [1])]
    else:
        constraint = []
    result = minimize(f, x0=x0, method='trust-constr',
                      bounds=bounds, constraints=constraint)
    vec = [s for s in result.x]
    return OrderedDict(zip(model_list, vec))


def get_sse(materials, benchmarks, model_score_dict=None):
    """
    Calculate the sum squared error for aggregated data
    weighted by the specified model scoring scheme against a set of benchmarks.

    Args:
        materials (`list` of `propnet.core.materials.Material`): list of evaluated materials containing
            symbols for benchmarking
        benchmarks (`list` of `dict`): list of dicts, keyed by Symbol or symbol name, containing benchmark data
            for each material in ``materials``.
        model_score_dict (dict): dict of weights as floats to apply to models, keyed
            by model name or Model object

    Returns:
        float: sum squared error over all the benchmarks

    """
    sse = 0
    for material, benchmark in zip(materials, benchmarks):
        for symbol, value in benchmark.items():
            agg = aggregate_quantities(material[symbol], model_score_dict)
            if np.isnan(agg.magnitude):
                continue
            sse += (agg.magnitude - benchmark[symbol]) ** 2
    return sse
