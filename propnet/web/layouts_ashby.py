import dash_core_components as dcc
import dash_html_components as html

from maggma.stores import MongoStore
from os import environ
from monty.serialization import loadfn

from propnet.symbols import DEFAULT_SYMBOLS

from dash.dependencies import Input, Output, State

from pydash import get

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
#_ensure_indices()

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
        #html.Label('Choose property to color by: '),
        #dcc.Dropdown(id='choose-y', options=[
        #    {'label': v.display_names[0], 'value': k} for k, v in scalar_symbols.items()
        #], value='volume_unit_cell'),
        dcc.Graph(id='ashby-graph')
    ])

    @app.callback(
        Output('ashby-graph', 'figure'),
        [Input('choose-x', 'value'),
         Input('choose-y', 'value')]
    )
    def update_graph(x_prop, y_prop):

        x_key = '{}.mean'.format(x_prop)
        y_key = '{}.mean'.format(y_prop)

        data = store.query(criteria={
            x_key: {"$exists": True},
            y_key: {"$exists": True}
        }, properties=['task_id', x_key, y_key])
        data = list(data)

        data = [
            {
                'x': [get(d, x_key) for d in data],
                'y': [get(d, y_key) for d in data],
                'text': [d['task_id'] for d in data],
                'opacity': 0.8,
                'mode': 'markers',
                'type': 'scattergl'
            }
        ]

        layout = {
            'yaxis': {'title': scalar_symbols[y_prop].display_names[0]},
            'xaxis': {'title': scalar_symbols[x_prop].display_names[0]}
        }

        return {'data': data, 'layout': layout}

    return layout
