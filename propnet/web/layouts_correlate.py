import dash_core_components as dcc
import dash_html_components as html

import numpy as np

from os import environ
from random import choice
from monty.serialization import loadfn

# noinspection PyUnresolvedReferences
import propnet.symbols
from propnet.core.registry import Registry

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

correlation_func_descriptions = {
            "mic": "Maximal information coefficient",
            "linlsq": "Linear least squares, R-squared",
            "theilsen": "Theil-Sen regression, R-squared",
            "ransac": "RANSAC regression",
            "pearson": "Pearson R correlation",
            "spearman": "Spearman R correlation"
        }


def violin_plot(correlation_func="mic"):

    path_lengths = sorted(
        store.query().distinct("shortest_path_length"), key=lambda x: (x is None, x)
    )

    props = {p: True for p in ["property_x", "property_y", "shortest_path_length",
                               "correlation", "n_points"]}
    props['_id'] = False

    docs = store.query(
        criteria={"correlation_func": correlation_func, "n_points": {"$ne": 0}},
        properties=props
    )

    # all_correlations = [d['correlation']
    #                     for d in store.query(criteria={"correlation_func": correlation_func, "n_points": {"$ne": 0}},
    #                                          properties=["correlation"])]
    # ymax = np.nanpercentile(all_correlations, 90)
    # ymin = np.nanpercentile(all_correlations, 10)

    points = {p: [] for p in path_lengths}
    for d in docs:
        points[d["shortest_path_length"]].append(d)
    data = []
    for p in path_lengths:
        points_p = points[p]
        if len(points_p) == 0:
            continue

        trace = {
            "type": "violin",
            "x": [str(p)] * len(points_p),
            "y": ["{:0.5f}".format(point['correlation'])
                  for point in points_p if point['correlation'] is not None],
            "customdata": [d for d in points_p],
            "name": str(p),
            "box": {"visible": True},
            "points": "all",
            "meanline": {"visible": False},
            "hoverinfo": "y",
        }
        data.append(trace)

    func_description = correlation_func_descriptions[correlation_func]

    layout = {
        "title": f"Correlation between properties based on {func_description} score",
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
        value="mic",
    )

    path_length_choice = dcc.Dropdown(
        id="path_length_choice",
        options=[],
        value="",
    )

    point_information = dcc.Markdown(id="point_info", children="""
##### Point information
No point selected
""")

    explain_text = dcc.Markdown("""
_Note: You may encounter some display issues with the mouseover information, particularly when
zooming or changing the axes value ranges. These issues originate within the external library 
used to create the violin plot, which is still in development. To restore the mouseover, try
isolating the plot you wish to explore by double-clicking its legend entry and/or setting the
axes to a wider display range._
""")

    @app.callback(
        Output("correlation_violin", "figure"),
        [Input("correlation_func_choice", "value")],
    )
    def regenerate_figure_for_new_correlation_func(correlation_func):
        if not correlation_func:
            raise PreventUpdate
        return violin_plot(correlation_func)

    @app.callback(
        Output("point_info", "children"),
        [Input("correlation_violin", "clickData")],
        [State("correlation_func_choice", "value")]
    )
    def populate_point_information(selected_points, correlation_func):
        if not selected_points:
            raise PreventUpdate

        target_data = selected_points['points'][0]['customdata']
        prop_x_name = Registry("symbols")[target_data['property_x']].display_names[0]
        prop_y_name = Registry("symbols")[target_data['property_y']].display_names[0]
        text = f"""
##### Point information
**x-axis property:** {prop_x_name}

**y-axis property:** {prop_y_name}

**correlation function used:** {correlation_func_descriptions[correlation_func]}

**correlation value:** {target_data['correlation']:0.5f}

**number of points tested:** {target_data['n_points']}
"""
        return text

    plot_display_layout = html.Div([
        correlation_func_choice,
        path_length_choice,
        graph],
        className="seven columns")

    info_layout = html.Div([point_information],
                           className="five columns")

    layout = html.Div([
        html.Div([plot_display_layout, info_layout],
                 className="row"),
        html.Div([explain_text],
                 className="row")
    ])

    return layout
