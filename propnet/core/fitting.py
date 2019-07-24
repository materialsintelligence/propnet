"""
This module contains code relevant to using fitting to improve
the aggregation process
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
    Simple method for aggregating a set of quantities

    Args:
        quantities:
        model_score_dict:

    Returns:

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
    Gets weight based on scoring scheme

    Args:
        quantity (Quantity): quantity for which to get weight
        model_score_dict ({str: float}): dictionary of model names to scores

    Returns:
        calculated weight for input quantity
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
    Fits a set of model scores to a set of benchmark data

    Args:
        materials ([Material]): list of evaluated materials containing
            symbols for benchmarking
        benchmarks ([{Symbol or str: float}]): list of dicts, keyed by Symbol
            or symbol name containing benchmark data for each material in ``materials``.
        models ([Model or str]): list of models which should have their
            scores adjusted in the aggregation weighting scheme
        init_scores ({str: float}): initial scores for minimization procedure.
            If unspecified, all scores are equal. Scores are normalized to sum of
            scores.
        constrain_sum (bool): True constrains the sum of weights to 1, False
            removes this constraint. Default: False (no constraint)

    Returns:
        {str: float} scores corresponding to those which minimize
            SSE for the benchmarked dataset

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
    Function to get the sum squared error of a set of benchmarks
    with aggregated data from the model scoring scheme above

    Args:
        materials ([Material]): list of materials to evaluate
        benchmarks ([{Symbol or str: float}]): list of benchmarks
            for each material
        model_score_dict ({str: float}): model score dictionary
            with scores for each model name

    Returns:
        (float): sum squared error over all the benchmarks

    """
    sse = 0
    for material, benchmark in zip(materials, benchmarks):
        for symbol, value in benchmark.items():
            agg = aggregate_quantities(material[symbol], model_score_dict)
            if np.isnan(agg.magnitude):
                continue
            sse += (agg.magnitude - benchmark[symbol]) ** 2
    return sse
