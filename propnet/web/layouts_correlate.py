import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt

import os
from monty.serialization import loadfn
from monty.json import MontyDecoder

# noinspection PyUnresolvedReferences
import propnet.symbols
from propnet.core.registry import Registry

from propnet.web.layouts_plot import scalar_symbols

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go

from pymatgen import MPRester

from pymongo.errors import ServerSelectionTimeoutError

import logging
import numpy as np

logger = logging.getLogger(__name__)
mpr = MPRester()


try:
    store_data = os.environ["PROPNET_CORRELATION_STORE_FILE"]
    if os.path.exists(store_data):
        # If store_data contains a file path
        store = loadfn(store_data)
    else:
        # Store data contains a json string
        store = MontyDecoder().decode(store_data)
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


def violin_plot(correlation_func=None):
    if correlation_func is None:
        return go.Figure()

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

    points = {p: [] for p in path_lengths}
    for d in docs:
        points[d["shortest_path_length"]].append(d)
    data = []
    bounds_test_func = correlation_func_info[correlation_func]['bounds']
    pos_idx = 0
    for p in path_lengths:
        points_p = points[p]
        if len(points_p) == 0:
            continue
        y_data = []
        custom_data = []

        for pt in points_p:
            corr = pt['correlation']
            if corr is not None and np.isfinite(corr) and bounds_test_func(corr):
                y_data.append("{:0.5f}".format(corr))
                custom_data.append(pt)

        x_data = [pos_idx] * len(y_data)
        if p is not None:
            name = f"{p} model"
            if p != 1:
                name += "s"
        else:
            name = "Not connected"
        trace = {
            "type": "violin",
            "x": x_data,
            "y": y_data,
            "customdata": custom_data,
            "name": name,
            "box": {"visible": True},
            "points": "all",
            "meanline": {"visible": False},
            "hoverinfo": "y",
        }
        data.append(trace)
        pos_idx += 1

    func_description = correlation_func_info[correlation_func]["name"]

    layout = {
        "title": f"Correlation between properties based on<br>{func_description} score",
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
        options=[{"label": v['name'], "value": k} for k, v in correlation_func_info.items()],
        value="mic",
        placeholder="Select correlation function"
    )

    explain_text = dcc.Markdown("""
##### How to read this plot

The plot above is called a "violin plot." The width of the plot shows the density of points that have
the value on the y-axis. Inscribed in the "violin" is a standard box-and-whisker plot, with lines 
indicating mean and quartile values. To the left of the violin plot are the actual data points used to
calculate the statistical data.

Each violin plot shows correlation scores for scalar property pairs separated by different numbers 
of models on the _propnet_ knowledge graph. Property pairs that are not currently connected by any path 
are shown on the far right.

Select the dropdown above the plot to choose between different correlation metrics. For certain metrics,
the data have been filtered to make the plot more readable.

To isolate a single violin plot, double-click its entry in the legend. You can hide/show plots by clicking
their entries on the legend. To find out more information about a point on the plot, click it and the
information will be populated on the right. Click "View the data plot" to show the two properties plotted
against each other in the "Plot" view.

_Note: If you encounter issues with the mouseover labels on the plot, try
isolating the plot you wish to explore by double-clicking its legend entry and/or setting the
axes to a wider display range. The graphing package is in development._
""")

    plot_display_layout = html.Div([
        correlation_func_choice,
        graph],
        className="seven columns")

    info_layout = html.Div(className="five columns", children=[
        dcc.Dropdown(id='choose-corr-x', options=[
            {'label': v.display_names[0], 'value': k} for k, v in
            scalar_symbols.items()
        ], placeholder="Select first property"),
        dcc.Dropdown(id='choose-corr-y', options=[
            {'label': v.display_names[0], 'value': k} for k, v in
            scalar_symbols.items()
        ], placeholder="Select second property"),
        html.Div(id="point_info",
                 children=[dcc.Markdown("""
##### Point information
No point selected
""")])
    ])

    point_clicked = dcc.Store(id='point_clicked', storage_type='memory',
                              data=False)

    layout = html.Div([
        html.Div([plot_display_layout, info_layout, point_clicked],
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
        Output("corr-table", "style_data_conditional"),
        [Input("correlation_violin", "figure")],
        [State("correlation_func_choice", "value"),
         State("point_clicked", "data")]
    )
    def highlight_table_row(_, selected_func, point_info_populated):
        if not point_info_populated:
            raise PreventUpdate
        table_highlight = [{
            'if': {'row_index': correlation_funcs.index(selected_func)},
            "backgroundColor": "#3D9970",
            'color': 'white'
        }]
        return table_highlight

    @app.callback(
        [Output("choose-corr-x", "value"),
         Output("choose-corr-y", "value")],
        [Input("correlation_violin", "clickData")]
    )
    def update_xy_selection(selected_points):
        if not selected_points:
            raise PreventUpdate
        target_data = selected_points['points'][0]['customdata']
        prop_x, prop_y = target_data['property_x'], target_data['property_y']

        return prop_x, prop_y

    @app.callback(
        [Output("point_info", "children"),
         Output("point_clicked", "data")],
        [Input("choose-corr-x", "value"),
         Input("choose-corr-y", "value")],
        [State("correlation_func_choice", "value")]
    )
    def populate_point_information(prop_x, prop_y, current_func):
        if not (prop_x and prop_y):
            raise PreventUpdate

        prop_x_name = Registry("symbols")[prop_x].display_names[0]
        prop_y_name = Registry("symbols")[prop_y].display_names[0]

        data = list(store.query(criteria={'property_x': prop_x,
                                          'property_y': prop_y}))

        path_length = data[0]['shortest_path_length']
        if path_length is None:
            path_text = "not connected"
        elif path_length == 0:
            path_text = "properties are the same"
        else:
            path_text = f"separated by {path_length} model"
            if path_length > 1:
                path_text += "s"
        point_text = dcc.Markdown(f"""
##### Point information
**x-axis property:** {prop_x_name}

**y-axis property:** {prop_y_name}

**distance apart on graph:** {path_text}

**number of data points:** {data[0]['n_points']}
""")

        # This ensures we know the ordering of the rows
        correlation_data = {
            d['correlation_func']:
                {'Correlation Function': correlation_func_info[d['correlation_func']]["name"],
                 'Correlation Value': f"{d['correlation']:0.5f}"}
            for d in data}
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
        link_to_plot = dcc.Link("View the data plot",
                                href=f'/plot?x={prop_x}&y={prop_y}')
        return [point_text, correlation_table, link_to_plot], True

    return layout
