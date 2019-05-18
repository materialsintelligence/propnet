import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt

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
from copy import deepcopy

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

correlation_funcs = list(store.query().distinct("correlation_func"))

correlation_func_info = {
            "mic": {"name": "Maximal information coefficient",
                    "bounds": lambda x: 0 <= round(x) <= 1},
            "linlsq": {"name": "Linear least squares, R-squared",
                       "bounds": lambda x: 0 <= round(x) <= 1},
            "theilsen": {"name": "Theil-Sen regression, R-squared",
                         "bounds": lambda x: -10 <= round(x) <= 1},    # Arbitrary lower bound to filter nonsense data
            "ransac": {"name": "RANSAC regression",
                       "bounds": lambda x: -10 <= round(x) <= 1},  # Arbitrary lower bound to filter nonsense data
            "pearson": {"name": "Pearson R correlation",
                        "bounds": lambda x: -1 <= round(x) <= 1},
            "spearman": {"name": "Spearman R correlation",
                         "bounds": lambda x: -1 <= round(x) <= 1}
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
    bounds_test_func = correlation_func_info[correlation_func]['bounds']
    for p in path_lengths:
        points_p = points[p]
        if len(points_p) == 0:
            continue
        y_data = []
        custom_data = []

        for pt in points_p:
            if pt['correlation'] is not None and bounds_test_func(pt['correlation']):
                y_data.append("{:0.5f}".format(pt['correlation']))
                custom_data.append(pt)

        x_data = [str(p)] * len(y_data)
        trace = {
            "type": "violin",
            "x": x_data,
            "y": y_data,
            "customdata": custom_data,
            "name": str(p),
            "box": {"visible": True},
            "points": "all",
            "meanline": {"visible": False},
            "hoverinfo": "y",
        }
        data.append(trace)

    func_description = correlation_func_info[correlation_func]["name"]

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

    explain_text = dcc.Markdown("""
_Note: You may encounter some display issues with the mouseover information, particularly when
zooming or changing the axes value ranges. These issues originate within the external library 
used to create the violin plot, which is still in development. To restore the mouseover, try
isolating the plot you wish to explore by double-clicking its legend entry and/or setting the
axes to a wider display range._
""")

    plot_display_layout = html.Div([
        correlation_func_choice,
        path_length_choice,
        graph],
        className="six columns")

    info_layout = html.Div(id="point_info", className="six columns",
                           children=[dcc.Markdown("""
##### Point information
No point selected
""")])

    layout = html.Div([
        html.Div([plot_display_layout, info_layout],
                 className="row"),
        html.Div([explain_text],
                 className="row")
    ])

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
    def populate_point_information(selected_points, current_func):
        if not selected_points:
            raise PreventUpdate

        target_data = selected_points['points'][0]['customdata']
        prop_x_name = Registry("symbols")[target_data['property_x']].display_names[0]
        prop_y_name = Registry("symbols")[target_data['property_y']].display_names[0]
        if target_data['shortest_path_length'] is None:
            path_text = "the properties not connected"
        elif target_data['shortest_path_length'] == 0:
            path_text = "the properties are the same"
        else:
            path_text = f"separated by {target_data['shortest_path_length']} model"
            if target_data['shortest_path_length'] > 1:
                path_text += "s"
        point_text = dcc.Markdown(f"""
##### Point information
**x-axis property:** {prop_x_name}

**y-axis property:** {prop_y_name}

**distance apart on graph:** {path_text}

**number of data points:** {target_data['n_points']}
""")
        query = store.query(criteria={'property_x': target_data['property_x'],
                                      'property_y': target_data['property_y']},
                            properties=["correlation_func", "correlation"])
        # This ensures we know the ordering of the rows
        correlation_data = {
            d['correlation_func']:
                {'Correlation Function': correlation_func_info[d['correlation_func']]["name"],
                 'Correlation Value': f"{d['correlation']:0.5f}"}
            for d in query}
        correlation_data = [correlation_data[func] for func in correlation_funcs]

        correlation_table = dt.DataTable(id='corr-table', data=correlation_data,
                                         columns=[{'id': val, 'name': val}
                                                  for val in ('Correlation Function', 'Correlation Value')],
                                         editable=False,
                                         style_data_conditional=[{
                                             'if': {'row_index': correlation_funcs.index(current_func)},
                                             "backgroundColor": "#3D9970",
                                             'color': 'white'
                                         }],
                                         style_cell={
                                             'font-family': 'HelveticaNeue',
                                             'text-align': 'left'
                                         },
                                         style_header={
                                             'fontWeight': 'bold',
                                             'font-family': 'HelveticaNeue',
                                             'text-align': 'left'
                                         })
        return [point_text, correlation_table]

    return layout
