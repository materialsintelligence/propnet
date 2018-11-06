import dash_core_components as dcc
import dash_html_components as html

from os import environ
from monty.serialization import loadfn

from propnet.symbols import DEFAULT_SYMBOLS

from dash.dependencies import Input, Output, State

from pydash import get

from pymatgen import MPRester
from pymatgen.util.string import unicodeify

mpr = MPRester()

store = loadfn(environ["PROPNET_STORE_FILE"])
store.connect()

cut_off = 100  # need at least this many available quantities for plot
scalar_symbols = {k: v for k, v in DEFAULT_SYMBOLS.items()
                  if (v.category == 'property' and v.shape == 1
                      and store.query(criteria={f'{k}.mean': {'$exists': True}}).count() > cut_off)}

# this is dependent on the schema format


def _ensure_indices():
    for property_name in scalar_symbols.keys():
        store.ensure_index(property_name)


def ashby_layout(app):

    layout = html.Div([
        html.Label('Choose property for x-axis: '),
        dcc.Dropdown(id='choose-x', options=[
            {'label': v.display_names[0], 'value': k} for k, v in scalar_symbols.items()
        ], value='atomic_density'),
        html.Label('Choose property for y-axis: '),
        dcc.Dropdown(id='choose-y', options=[
            {'label': v.display_names[0], 'value': k} for k, v in scalar_symbols.items()
        ], value='volume_unit_cell'),
        html.Div([
            html.Div([dcc.Graph(id='ashby-graph', config={'displayModeBar': False})], className='eight columns'),
            html.Div([html.Br(), html.Br(),
                      dcc.Markdown(id='point-detail', children="Click on a point for more information on that material.")],
                     className='four columns')
        ], className='row')
    ])

    @app.callback(
        Output('point-detail', 'children'),
        [Input('ashby-graph', 'clickData')],
        [State('choose-x', 'value'),
         State('choose-y', 'value')]
    )
    def update_info_box(clickData, x_prop, y_prop):

        point = clickData['points'][0]
        mpid = point['text']
        x = point['x']
        y = point['y']

        s = mpr.get_structure_by_material_id(mpid)
        formula = unicodeify(s.composition.reduced_formula)

        return f"""
        
### {formula}
##### [{mpid}](https://materialsproject.org/materials/{mpid})
        
x = {x:.2f} {scalar_symbols[x_prop].unit_as_string}

y = {y:.2f} {scalar_symbols[y_prop].unit_as_string}
        """

    @app.callback(
        Output('ashby-graph', 'figure'),
        [Input('choose-x', 'value'),
         Input('choose-y', 'value')]
    )
    def update_graph(x_prop, y_prop):

        x_key = '{}.mean'.format(x_prop)
        y_key = '{}.mean'.format(y_prop)
        x_std_dev = '{}.std_dev'.format(x_prop)
        y_std_dev = '{}.std_dev'.format(y_prop)

        data = store.query(criteria={
            x_key: {"$exists": True},
            y_key: {"$exists": True}
        }, properties=['task_id', x_key, y_key])
        data = list(data)

        data = [
            {
                'x': [get(d, x_key) for d in data],
                'y': [get(d, y_key) for d in data],
                #'error_x': {'type': 'data', 'array': [get(d, x_std_dev, 0) for d in data], 'visible': True},
                #'error_y': {'type': 'data', 'array': [get(d, y_std_dev, 0) for d in data], 'visible': True},
                'text': [d['task_id'] for d in data],
                'mode': 'markers',
                'marker': {'size': 3},
                'type': 'scattergl'
            }
        ]

        x_title = "{} / {}".format(scalar_symbols[x_prop].display_names[0],
                                   scalar_symbols[x_prop].unit_as_string)

        y_title = "{} / {}".format(scalar_symbols[y_prop].display_names[0],
                                   scalar_symbols[y_prop].unit_as_string)

        layout = {
            'yaxis': {'title': y_title, 'showgrid': False},
            'xaxis': {'title': x_title, 'showgrid': False},
            'hovermode': 'closest'
        }

        return {'data': data, 'layout': layout}

    return layout
