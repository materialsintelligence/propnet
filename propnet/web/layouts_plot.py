import dash_core_components as dcc
import dash_html_components as html
from dash import callback_context as ctx
import plotly.graph_objs as go

import numpy as np

from os import environ
from random import choice
from monty.serialization import loadfn

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from pydash import get, set_

from pymatgen import MPRester
from pymatgen.util.string import unicodeify

# noinspection PyUnresolvedReferences
import propnet.symbols
from propnet.core.registry import Registry

from pymongo.errors import ServerSelectionTimeoutError

mpr = MPRester()

try:
    store = loadfn(environ["PROPNET_STORE_FILE"])
    store.connect()
except (ServerSelectionTimeoutError, KeyError):
    from maggma.stores import MemoryStore
    store = MemoryStore()
    store.connect()
    # layout won't work if database is down, but at least web app will stay up
    scalar_symbols = {k: v for k, v in Registry("symbols").items()
                      if (v.category == 'property' and v.shape == 1)}
    warning_layout = html.Div('No database connection could be established.',
                              style={'font-family': 'monospace',
                                     'color': 'rgb(211, 84, 0)',
                                     'text-align': 'left',
                                     'font-size': '1.2em'})
else:
    cut_off = 100  # need at least this many available quantities for plot
    """
    scalar_symbols = {k: v for k, v in Registry("symbols").items()
                      if (v.category == 'property' and v.shape == 1
                          and store.query(
                              criteria={f'{k}.mean': {'$exists': True}}).count() > cut_off)}
    """
    scalar_symbols = {
        k: v for k, v in Registry("symbols").items()
        if (v.category == 'property' and v.shape == 1
            and len(store.query(criteria={'symbol_type': k}).distinct("material_key")) > cut_off)}
    warning_layout = html.Div()


# this is dependent on the database schema
def _ensure_indices():
    for property_name in scalar_symbols.keys():
        store.ensure_index(property_name)


def get_plot_layout(props=None):
    prop_x = None
    prop_y = None
    prop_z = None
    z_enabled = False
    create_plot = False
    if props:
        if props.get('x') is not None:
            prop_x = props['x']
        if props.get('y') is not None:
            prop_y = props['y']
        if props.get('z') is not None:
            prop_z = props['z']
            z_enabled = True

        if prop_x and prop_y:
            create_plot = True

    plot_data = None
    graph_config = dict(id='ashby-graph',
                        config={'displayModeBar': False},
                        style={'height': 600},
                        figure={'data': [], 'layout': {'margin': {'t': 20}}})
    if create_plot:
        plot_data = get_graph_data(prop_x, prop_y, prop_z, None)
        if plot_data['query_success']:
            graph_figure = get_graph_figure(plot_data, ['zoom'],
                                            ['enable-z'] if z_enabled else [],
                                            [], [])
            graph_config['figure'] = graph_figure
            n_points = len(plot_data['x'])
        else:
            n_points = 0
        count_text = f"There are {n_points} data points in the pre-built propnet " \
            f"database that matches these criteria."
    else:
        count_text = ""

    graph_layout = dcc.Graph(**graph_config)
    graph_store = dcc.Store(id='plot-data', storage_type='memory',
                            data=plot_data)

    controls_layout = html.Div([
        html.Label('Choose property for x-axis: '),
        dcc.Dropdown(id='choose-x', options=[
            {'label': v.display_names[0], 'value': k} for k, v in
            scalar_symbols.items()
        ], value=prop_x),
        html.Label('Choose property for y-axis: '),
        dcc.Dropdown(id='choose-y', options=[
            {'label': v.display_names[0], 'value': k} for k, v in
            scalar_symbols.items()
        ], value=prop_y),
        dcc.Checklist(id='zoom', options=[
            {'label': 'Zoom to 90% percentile', 'value': 'zoom'}],
                      values=['zoom'], labelStyle={'display': 'inline-block'}),
        dcc.Checklist(id='enable-color',
                      options=[{'label': 'Enable color scale',
                                'value': 'enable-color'}],
                      values=[], labelStyle={'display': 'inline-block'}),
        html.Div([
            html.Label('Choose property for color scale: '),
            dcc.Dropdown(id='choose-color', options=[
                {'label': v.display_names[0], 'value': k} for k, v in
                scalar_symbols.items()
            ], value=[]),
            html.Label('Set range for color scale: '),
            dcc.RangeSlider(id='color-scale-range', step=0.01),
            html.Br()
        ], style={'display': 'none'}, id='color-scale-controls'),
        dcc.Checklist(id='enable-z',
                      options=[{'label': 'Enable z-axis', 'value': 'enable-z'}],
                      values=['enable-z'] if z_enabled else [],
                      labelStyle={'display': 'inline-block'}),
        html.Div([
            html.Label('Choose property for z-axis: '),
            dcc.Dropdown(id='choose-z', options=[
                {'label': v.display_names[0], 'value': k} for k, v in
                scalar_symbols.items()
            ], value=prop_z)], style={'display': 'none'} if not z_enabled else {},
            id='z-axis-controls'),
        html.Br(),
        html.Div(id='query-counter', children=count_text)
    ])

    detail_layout = html.Div([html.Br(), html.Br(),
                              dcc.Markdown(id='point-detail',
                                           children="Click on a point for more information on that material."),
                              html.Div(
                                  #html.Details(html.Summary(
                                  #    'Provenance for x datapoint')),
                                  #html.Details(html.Summary(
                                  #    'Provenance for y datapoint'))
                              )])

    description_layout = dcc.Markdown(
        "Plot materials properties pre-computed by propnet and seeded with "
        "input data from the [Materials Project](https://materialsproject.org) (additional data source integration "
        "is pending).")

    layout = html.Div([
        html.Div([description_layout, warning_layout, html.Br()],
                 className="row"),
        html.Div([html.Div([graph_layout, graph_store], className="seven columns"),
                  html.Div([controls_layout], className="five columns")],
                 className="row"),
        html.Div([detail_layout], className="row")
    ])

    return layout


def define_plot_callbacks(app):
    @app.callback(
        Output('z-axis-controls', 'style'),
        [Input('enable-z', 'values')]
    )
    def show_z_axis_controls(enable_z):
        if enable_z:
            return {}
        else:
            return {'display': 'none'}

    @app.callback(
        Output('color-scale-controls', 'style'),
        [Input('enable-color', 'values')]
    )
    def show_color_controls(enable_color):
        if enable_color:
            return {}
        else:
            return {'display': 'none'}

    @app.callback(
        Output('point-detail', 'children'),
        [Input('ashby-graph', 'clickData')],
        [State('choose-x', 'value'),
         State('choose-y', 'value'),
         State('choose-z', 'value'),
         State('choose-color', 'value')]
    )
    def update_info_box(clickData, x_prop, y_prop, z_prop, color_prop):

        if clickData is None:
            raise PreventUpdate

        point = clickData['points'][0]
        mpid = point['text']
        x = point['x']
        y = point['y']
        print(point)

        s = mpr.get_structure_by_material_id(mpid)
        formula = unicodeify(s.composition.reduced_formula)

        info = f"""

### {formula}
##### [{mpid}](https://materialsproject.org/materials/{mpid})

x = {x:.2f} {scalar_symbols[x_prop].unit_as_string}

y = {y:.2f} {scalar_symbols[y_prop].unit_as_string}

"""
        if 'z' in point:
            z = point['z']
            info += f"z = {z:.2f} {scalar_symbols[z_prop].unit_as_string}"

        if 'marker.color' in point:
            c = point['marker.color']
            info += f"c = {c:.2f} {scalar_symbols[color_prop].unit_as_string}"

        return info

    @app.callback(
        [Output('plot-data', 'data'),
         Output('query-counter', 'children'),
         Output('color-scale-range', 'min'),
         Output('color-scale-range', 'max'),
         Output('color-scale-range', 'marks'),
         Output('color-scale-range', 'value')],
        [Input('choose-x', 'value'),
         Input('choose-y', 'value'),
         Input('enable-z', 'values'),
         Input('choose-z', 'value'),
         Input('enable-color', 'values'),
         Input('choose-color', 'value')],
        [State('plot-data', 'data'),
         State('zoom', 'values'),
         State('color-scale-range', 'min'),
         State('color-scale-range', 'max'),
         State('color-scale-range', 'marks'),
         State('color-scale-range', 'value'),
         State('query-counter', 'children')]
    )
    def update_graph_data(x_prop, y_prop,
                          enable_z, z_prop,
                          enable_color, color_prop,
                          data, zoom,
                          slider_min, slider_max, slider_marks, slider_values,
                          count_text):

        if not x_prop or not y_prop or \
                (not z_prop and enable_z) or (not color_prop and enable_color):
            raise PreventUpdate

        last_clicked = ctx.triggered[0]['prop_id'].rsplit('.', 1)[0]

        if last_clicked == 'enable-color' and not color_prop:
            raise PreventUpdate

        if last_clicked == 'enable-z' and not z_prop:
            raise PreventUpdate

        if last_clicked in ('choose-x', 'choose-y', 'choose-z', 'choose-color'):
            new_data = get_graph_data(x_prop, y_prop,
                                      z_prop if enable_z else None,
                                      color_prop if enable_color else None)

            if new_data.get('query_success'):
                if enable_color and color_prop:
                    c = new_data['color']
                    slider_min = np.min(c)
                    slider_max = np.max(c)
                    units = scalar_symbols[color_prop].unit_as_string
                    slider_marks = {
                        np.min(c): f"{slider_min:.2f} {units}",
                        np.percentile(c, 10): "",  # f"{np.percentile(c, 10):.2f} {units}",
                        np.percentile(c, 50): "",  # f"{np.percentile(c, 50):.2f} {units}",
                        np.percentile(c, 90): "",  # f"{np.percentile(c, 90):.2f} {units}",
                        np.max(c): f"{slider_max:.2f} {units}",
                    }
                    slider_values = get_color_range(c, zoom)

                n_points = len(new_data['x'])
            else:
                n_points = 0
                new_data = data
                new_data['query_success'] = False

            count_text = f"There are {n_points} data points in the pre-built propnet " \
                f"database that matches these criteria."
        else:
            new_data = data

        return new_data, count_text, slider_min, slider_max, slider_marks, slider_values

    @app.callback(
        Output('ashby-graph', 'figure'),
        [Input('plot-data', 'data'),
         Input('color-scale-range', 'value'),
         Input('zoom', 'values')],
        [State('enable-z', 'values'),
         State('enable-color', 'values')]
    )
    def update_data_dependencies(data, color_range, zoom, enable_z, enable_color):
        if not data or not data.get('query_success'):
            raise PreventUpdate

        figure = get_graph_figure(data, zoom, enable_z,
                                  enable_color, color_range)

        return figure


def get_graph_data(x_prop, y_prop, z_prop, color_prop):
    props_to_get = [prop for prop in (x_prop, y_prop, z_prop, color_prop)
                    if prop is not None]
    fields = [f'{prop}.mean' for prop in props_to_get]
    criteria = {field: {'$exists': True}
                for field in fields}
    properties = fields + ['task_id']

    from propnet.dbtools.correlation import CorrelationBuilder
    data = CorrelationBuilder.get_data_from_quantity_db(store, *props_to_get,
                                                        include_id=True)
    # query = store.query(criteria=criteria, properties=properties)

    if len(data[x_prop]) == 0:
        return {'query_success': False}

    data = {
        'query_success': True,
        'x': data[x_prop],
        'x_name': x_prop,
        'y': data[y_prop],
        'y_name': y_prop,
        'mpids': data['_id'],
        'zoom': None,
        'color_range': None
    }
    if z_prop:
        data.update({'z': data[z_prop], 'z_name': z_prop, 'z_enabled': True})
    if color_prop:
        data.update({'color': data[color_prop], 'color_name': color_prop, 'color_enabled': True})

    return data


def get_graph_figure(data, zoom, enable_z, enable_color, color_range):
    x_prop = data['x_name']
    x = data['x']
    y_prop = data['y_name']
    y = data['y']
    mpids = data['mpids']

    trace = {
        'x': x,
        'y': y,
        'text': mpids,
        'mode': 'markers',
        'marker': {'size': 5},
    }

    if enable_color:
        c = data['color']
        color_prop = data['color_name']
        color_title = "{} / {}".format(scalar_symbols[color_prop].display_names[0],
                                       scalar_symbols[color_prop].unit_as_string)
        if not color_range:
            color_range = get_color_range(c, zoom)
        set_(trace, 'marker.color', c)
        set_(trace, 'marker.colorscale', 'Viridis')
        set_(trace, 'marker.showscale', True)
        set_(trace, 'marker.colorbar', {'title': color_title})
        set_(trace, 'marker.cmin', color_range[0])
        set_(trace, 'marker.cmax', color_range[1])

    x_title = "{} / {}".format(scalar_symbols[x_prop].display_names[0],
                               scalar_symbols[x_prop].unit_as_string)

    y_title = "{} / {}".format(scalar_symbols[y_prop].display_names[0],
                               scalar_symbols[y_prop].unit_as_string)

    axes_layout = {
        'xaxis': {'title': x_title, 'showgrid': True, 'showline': True,
                  'zeroline': False},
        'yaxis': {'title': y_title, 'showgrid': True, 'showline': True,
                  'zeroline': False}
    }

    if zoom:
        set_(axes_layout, 'xaxis.range', [np.percentile(x, 10),
                                          np.percentile(x, 90)])
        set_(axes_layout, 'yaxis.range', [np.percentile(y, 10),
                                          np.percentile(y, 90)])
    layout = {
        'hovermode': 'closest',
        'margin': {'t': 20}
    }

    if enable_z:
        z_prop = data['z_name']
        z = data['z']
        trace['z'] = z

        z_title = "{} / {}".format(scalar_symbols[z_prop].display_names[0],
                                   scalar_symbols[z_prop].unit_as_string)

        set_(axes_layout, 'zaxis', {'title': z_title, 'showgrid': True, 'showline': True,
                                     'zeroline': False})

        if zoom:
            set_(axes_layout, 'zaxis.range', [np.percentile(z, 10),
                                              np.percentile(z, 90)])
        scatter = go.Scatter3d(**trace)
        layout['scene'] = axes_layout
    else:
        scatter = go.Scattergl(**trace)
        layout.update(axes_layout)

    return {'data': [scatter], 'layout': go.Layout(**layout)}


def get_color_range(c, zoom):
    if zoom:
        color_range = [np.percentile(c, 10), np.percentile(c, 90)]
    else:
        color_range = [np.min(c), np.max(c)]

    return color_range
