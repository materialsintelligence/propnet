import dash_core_components as dcc
import dash_html_components as html

import numpy as np

from os import environ
from random import choice
from monty.serialization import loadfn

from propnet.symbols import DEFAULT_SYMBOLS

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go
from pydash import get

from pymatgen import MPRester
from pymatgen.util.string import unicodeify

mpr = MPRester()

from pymongo.errors import ServerSelectionTimeoutError

store = loadfn(
    environ.get("PROPNET_CORRELATION_STORE_FILE", "propnet_correlation_store.json")
)

try:
    store.connect()
except ServerSelectionTimeoutError:
    # layout won't work if database is down, but at least web app will stay up
    pass


correlation_funcs = store.query().distinct("correlation_func")


def violin_plot(correlation_func="mic"):
    path_lengths = sorted(
        store.query().distinct("shortest_path_length"), key=lambda x: (x is None, x)
    )
    properties = store.query().distinct("property_x")

    docs = store.query(
        criteria={"correlation_func": correlation_func, "n_points": {"$ne": 0}},
        properties=["property_x", "property_y", "shortest_path_length", "correlation"],
    )

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
            "x": [p if isinstance(p, int) else 8]*len(points_p),
            "y": [point[0] for point in points_p],
            "text": [point[1] for point in points_p],
            "name": str(p),
            "box": {"visible": True},
            "meanline": {"visible": True},
        }
        data.append(trace)

    layout = {
        "title": f"Correlation between properties based on {correlation_func} score",
        "yaxis": {"zeroline": False, "showgrid": False, "title": "Correlation score"},
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
        value='mic',
    )

    @app.callback(
        Output('correlation_violin', 'figure'),
        [Input('correlation_func_choice', 'value')]
    )
    def regenerate_figure_for_new_correlation_func(correlation_func):
        if not correlation_func:
            raise PreventUpdate
        return violin_plot(correlation_func)

    layout = html.Div([correlation_func_choice, graph])

    return layout
