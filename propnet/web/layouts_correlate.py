import dash_core_components as dcc
import dash_html_components as html

import numpy as np

from os import environ
from random import choice
from monty.serialization import loadfn

# noinspection PyUnresolvedReferences
import propnet.symbols

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go
from pydash import get

from pymatgen import MPRester
from pymatgen.util.string import unicodeify

from pymongo.errors import ServerSelectionTimeoutError

import logging

logger = logging.getLogger(__name__)
mpr = MPRester()

try:
    store = loadfn(
        environ["PROPNET_CORRELATION_STORE_FILE"])
    store.connect()
except (ServerSelectionTimeoutError, KeyError, FileNotFoundError) as ex:
    if isinstance(ex, ServerSelectionTimeoutError):
        logger.warning("Unable to connect to propnet correlation db!")
    if isinstance(ex, KeyError):
        logger.warning("PROPNET_CORRELATION_STORE_FILE var not set!")
    if isinstance(ex, FileNotFoundError):
        logger.warning("File specified in PROPNET_CORRELATION_STORE_FILE not found!")
    from maggma.stores import MemoryStore
    store = MemoryStore()
    store.connect()
    # layout won't work if database is down, but at least web app will stay up

correlation_funcs = store.query().distinct("correlation_func")

def violin_plot(correlation_func="mic"):

    path_lengths = sorted(
        store.query().distinct("shortest_path_length"), key=lambda x: (x is None, x)
    )

    docs = store.query(
        criteria={"correlation_func": correlation_func, "n_points": {"$ne": 0}},
        properties=["property_x", "property_y", "shortest_path_length", "correlation"],
    )

    all_correlations = [d['correlation']
                        for d in store.query(criteria={"correlation_func": correlation_func, "n_points": {"$ne": 0}},
                                             properties=["correlation"])]
    ymax = np.nanpercentile(all_correlations, 90)
    ymin = np.nanpercentile(all_correlations, 10)

    points = {p: [] for p in path_lengths}
    for d in docs:
        points[d["shortest_path_length"]].append(
            (d["correlation"], f"{d['property_x']}-{d['property_y']}")
        )

    data = []
    for p in path_lengths:
        points_p = points[p]

        trace = {
            "type": "violin",
            "x": [str(p)] * len(points_p),
            "y": ["{:0.5f}".format(point[0]) for point in points_p],
            "text": [point[1] for point in points_p],
            "name": str(p),
            "box": {"visible": True},
            "points": "all",
            "meanline": {"visible": True},
            "hoverinfo": "y+text+name",
        }
        data.append(trace)

    layout = {
        "title": f"Correlation between properties based on {correlation_func} score",
        "yaxis": {"zeroline": False, "showgrid": False, "title": "Correlation score",
                  "range": [ymin, ymax]},
        "xaxis": {"showticklabels": False}
    }

    return go.Figure(data=data, layout=layout)


def correlate_layout(app):

    graph = dcc.Graph(
        figure=violin_plot(),
        style={"height": 600},
        config={"showLink": False, "displayModeBar": False},
        id="correlation_violin",
    )

    correlation_func_choice = dcc.Dropdown(
        id="correlation_func_choice",
        options=[{"label": f, "value": f} for f in correlation_funcs],
        value="mic",
    )

    @app.callback(
        Output("correlation_violin", "figure"),
        [Input("correlation_func_choice", "value")],
    )
    def regenerate_figure_for_new_correlation_func(correlation_func):
        if not correlation_func:
            raise PreventUpdate
        return violin_plot(correlation_func)

    layout = html.Div([correlation_func_choice, graph])

    return layout
