import dash_core_components as dcc
import dash_html_components as html

import numpy as np

from os import environ
from random import choice
from monty.serialization import loadfn

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from pydash import get

from pymatgen import MPRester
from pymatgen.util.string import unicodeify

# noinspection PyUnresolvedReferences
import propnet.symbols
from propnet.core.registry import Registry

mpr = MPRester()

from pymongo.errors import ServerSelectionTimeoutError

try:
    store = loadfn(environ["PROPNET_STORE_FILE"])
    store.connect()
except (ServerSelectionTimeoutError, KeyError):
    from maggma.stores import MemoryStore
    store = MemoryStore()
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
    scalar_symbols = {k: v for k, v in Registry("symbols").items()
                      if (v.category == 'property' and v.shape == 1
                          and store.query(
                              criteria={f'{k}.mean': {'$exists': True}}).count() > cut_off)}
    warning_layout = html.Div()


# this is dependent on the database schema
def _ensure_indices():
    for property_name in scalar_symbols.keys():
        store.ensure_index(property_name)


DEFAULT_X = 'band_gap'
DEFAULT_Y = 'volume_unit_cell'
DEFAULT_Z = 'atomic_density'
DEFAULT_COLOR = 'energy_above_hull'

def plot_layout(app):

    graph_layout = dcc.Graph(id='ashby-graph', config={'displayModeBar': False})

    controls_layout = html.Div([
        html.Label('Choose property for x-axis: '),
        dcc.Dropdown(id='choose-x', options=[
            {'label': v.display_names[0], 'value': k} for k, v in
            scalar_symbols.items()
        ], value=DEFAULT_X),
        html.Label('Choose property for y-axis: '),
        dcc.Dropdown(id='choose-y', options=[
            {'label': v.display_names[0], 'value': k} for k, v in
            scalar_symbols.items()
        ], value=DEFAULT_Y),
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
            ], value=DEFAULT_COLOR),
            html.Label('Set range for color scale: '),
            dcc.RangeSlider(id='color-scale-range', step=0.01),
            html.Br()
        ], style={'display': 'none'}, id='color-scale-controls'),
        dcc.Checklist(id='enable-z',
                      options=[{'label': 'Enable z-axis', 'value': 'enable-z'}],
                      values=[], labelStyle={'display': 'inline-block'}),
        html.Div([
            html.Label('Choose property for z-axis: '),
            dcc.Dropdown(id='choose-z', options=[
                {'label': v.display_names[0], 'value': k} for k, v in
                scalar_symbols.items()
            ], value=DEFAULT_Z)], style={'display': 'none'},
            id='z-axis-controls'),
        html.Br(),
        html.Div(id='query-counter')
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
        html.Div([html.Div([graph_layout], className="seven columns"),
                  html.Div([controls_layout], className="five columns")],
                 className="row"),
        html.Div([detail_layout], className="row")
    ])

    @app.callback(
        Output('color-scale-range', 'value'),
        [Input('choose-color', 'value'),
         Input('zoom', 'values')]
    )
    def update_range_slider_values(color_prop, zoom):

        color_key = '{}.mean'.format(color_prop)
        data = store.query(criteria={color_key: {"$exists": True}},
                           properties=[color_key])
        data = list(data)

        if not data:
            raise PreventUpdate

        c = [get(d, color_key) for d in data]

        if zoom:
            color_range = [np.percentile(c, 10), np.percentile(c, 90)]
        else:
            color_range = [np.min(c), np.max(c)]

        return color_range

    @app.callback(
        Output('color-scale-range', 'marks'),
        [Input('choose-color', 'value')]
    )
    def update_range_slider_marks(color_prop):

        color_key = '{}.mean'.format(color_prop)
        data = store.query(criteria={color_key: {"$exists": True}},
                           properties=[color_key])
        data = list(data)

        if not data:
            raise PreventUpdate

        c = [get(d, color_key) for d in data]
        units = scalar_symbols[color_prop].unit_as_string

        marks = {
            np.min(c): f"{np.min(c):.2f} {units}",
            np.percentile(c, 10): "",  # f"{np.percentile(c, 10):.2f} {units}",
            np.percentile(c, 50): "",  # f"{np.percentile(c, 50):.2f} {units}",
            np.percentile(c, 90): "",  # f"{np.percentile(c, 90):.2f} {units}",
            np.max(c): f"{np.max(c):.2f} {units}",
        }

        return marks

    @app.callback(
        Output('color-scale-range', 'min'),
        [Input('choose-color', 'value')]
    )
    def update_range_slider_marks(color_prop):

        color_key = '{}.mean'.format(color_prop)
        data = store.query(criteria={color_key: {"$exists": True}},
                           properties=[color_key])
        data = list(data)

        if not data:
            raise PreventUpdate

        c = [get(d, color_key) for d in data]
        return np.min(c)

    @app.callback(
        Output('color-scale-range', 'max'),
        [Input('choose-color', 'value')]
    )
    def update_range_slider_marks(color_prop):

        color_key = '{}.mean'.format(color_prop)
        data = store.query(criteria={color_key: {"$exists": True}},
                           properties=[color_key])
        data = list(data)

        if not data:
            raise PreventUpdate

        c = [get(d, color_key) for d in data]
        return np.max(c)


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

# {store.query_one(criteria={'task_id': mpid}, properties=[x_prop, y_prop])}

    @app.callback(
        Output('query-counter', 'children'),
        [Input('choose-x', 'value'),
         Input('choose-y', 'value'),
         Input('enable-z', 'values'),
         Input('enable-color', 'values'),
         Input('choose-z', 'value'),
         Input('choose-color', 'value')]
    )
    def update_graph_description(x_prop, y_prop,
                                 enable_z, enable_color, z_prop, color_prop):

        x_key = '{}.mean'.format(x_prop)
        y_key = '{}.mean'.format(y_prop)
        z_key = '{}.mean'.format(z_prop)
        color_key = '{}.mean'.format(color_prop)

        criteria = {
            x_key: {"$exists": True},
            y_key: {"$exists": True}
        }

        if enable_z:
            criteria[z_key] = {"$exists": True}

        if enable_color:
            criteria[color_key] = {"$exists": True}

        count = store.query(criteria=criteria, properties=[]).count()

        return f"There are {count} data points in the pre-built propnet " \
               f"database that matches these criteria."

    @app.callback(
        Output('ashby-graph', 'figure'),
        [Input('choose-x', 'value'),
         Input('choose-y', 'value'),
         Input('zoom', 'values'),
         Input('enable-z', 'values'),
         Input('enable-color', 'values'),
         Input('choose-z', 'value'),
         Input('choose-color', 'value'),
         Input('color-scale-range', 'value')]
    )
    def update_graph(x_prop, y_prop, zoom,
                     enable_z, enable_color, z_prop, color_prop, color_range):

        x_key = '{}.mean'.format(x_prop)
        y_key = '{}.mean'.format(y_prop)
        z_key = '{}.mean'.format(z_prop)
        color_key = '{}.mean'.format(color_prop)
        x_std_dev = '{}.std_dev'.format(x_prop)
        y_std_dev = '{}.std_dev'.format(y_prop)
        z_std_dev = '{}.std_dev'.format(z_prop)

        criteria = {
            x_key: {"$exists": True},
            y_key: {"$exists": True}
        }

        properties = ['task_id', x_key, y_key]

        if enable_z:
            criteria[z_key] = {"$exists": True}
            properties.append(z_key)

        if enable_color:
            criteria[color_key] = {"$exists": True}
            properties.append(color_key)

        data = store.query(criteria=criteria, properties=properties)
        data = list(data)

        if enable_z:

            x = [get(d, x_key) for d in data]
            y = [get(d, y_key) for d in data]
            z = [get(d, z_key) for d in data]

            traces = [
                {
                    'x': x,
                    'y': y,
                    'z': z,
                    # 'error_x': {'type': 'data', 'array': [get(d, x_std_dev, 0) for d in data], 'visible': True},
                    # 'error_y': {'type': 'data', 'array': [get(d, y_std_dev, 0) for d in data], 'visible': True},
                    'text': [d['task_id'] for d in data],
                    'mode': 'markers',
                    'marker': {'size': 5},
                    'type': 'scatter3d'
                }
            ]

            if enable_color:

                c = [get(d, color_key) for d in data]
                color_title = "{} / {}".format(scalar_symbols[color_prop].display_names[0],
                                       scalar_symbols[color_prop].unit_as_string)

                traces[0]['marker']['color'] = c
                traces[0]['marker']['colorscale'] = 'Viridis'
                traces[0]['marker']['showscale'] = True
                traces[0]['marker']['colorbar'] = {'title': color_title}
                traces[0]['marker']['cmin'] = color_range[0]
                traces[0]['marker']['cmax'] = color_range[1]


            x_title = "{} / {}".format(scalar_symbols[x_prop].display_names[0],
                                       scalar_symbols[x_prop].unit_as_string)

            y_title = "{} / {}".format(scalar_symbols[y_prop].display_names[0],
                                       scalar_symbols[y_prop].unit_as_string)

            z_title = "{} / {}".format(scalar_symbols[z_prop].display_names[0],
                                       scalar_symbols[z_prop].unit_as_string)

            layout = {
                'yaxis': {'title': y_title, 'showgrid': True, 'showline': True,
                          'zeroline': False},
                'xaxis': {'title': x_title, 'showgrid': True, 'showline': True,
                          'zeroline': False},
                'zaxis': {'title': z_title, 'showgrid': True, 'showline': True,
                          'zeroline': False},
                'hovermode': 'closest'
            }

            if zoom:
                layout['xaxis']['range'] = [np.percentile(x, 10),
                                            np.percentile(x, 90)]
                layout['yaxis']['range'] = [np.percentile(y, 10),
                                            np.percentile(y, 90)]
                layout['zaxis']['range'] = [np.percentile(z, 10),
                                            np.percentile(z, 90)]


        else:

            x = [get(d, x_key) for d in data]
            y = [get(d, y_key) for d in data]

            traces = [
                {
                    'x': x,
                    'y': y,
                    # 'error_x': {'type': 'data', 'array': [get(d, x_std_dev, 0) for d in data], 'visible': True},
                    # 'error_y': {'type': 'data', 'array': [get(d, y_std_dev, 0) for d in data], 'visible': True},
                    'text': [d['task_id'] for d in data],
                    'mode': 'markers',
                    'marker': {'size': 5},
                    'type': 'scattergl'
                }
            ]

            if enable_color:

                c = [get(d, color_key) for d in data]
                color_title = "{} / {}".format(scalar_symbols[color_prop].display_names[0],
                                       scalar_symbols[color_prop].unit_as_string)

                traces[0]['marker']['color'] = c
                traces[0]['marker']['colorscale'] = 'Viridis'
                traces[0]['marker']['showscale'] = True
                traces[0]['marker']['colorbar'] = {'title': color_title}
                traces[0]['marker']['cmin'] = color_range[0]
                traces[0]['marker']['cmax'] = color_range[1]


            x_title = "{} / {}".format(scalar_symbols[x_prop].display_names[0],
                                       scalar_symbols[x_prop].unit_as_string)

            y_title = "{} / {}".format(scalar_symbols[y_prop].display_names[0],
                                       scalar_symbols[y_prop].unit_as_string)

            layout = {
                'yaxis': {'title': y_title, 'showgrid': False, 'showline': True,
                          'zeroline': False},
                'xaxis': {'title': x_title, 'showgrid': False, 'showline': True,
                          'zeroline': False},
                'hovermode': 'closest'
            }

            if zoom:
                layout['xaxis']['range'] = [np.percentile(x, 10),
                                            np.percentile(x, 90)]
                layout['yaxis']['range'] = [np.percentile(y, 10),
                                            np.percentile(y, 90)]

        return {'data': traces, 'layout': layout}

    return layout
