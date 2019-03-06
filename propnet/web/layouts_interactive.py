import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt

from crystal_toolkit import GraphComponent
from propnet.web.utils import graph_conversion, AESTHETICS

import json
from monty.json import MontyEncoder, MontyDecoder

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from collections import OrderedDict

# noinspection PyUnresolvedReferences
import propnet.symbols
from propnet.core.registry import Registry

from propnet import ureg, logger
from propnet.core.quantity import QuantityFactory
from propnet.core.materials import Material
from propnet.core.graph import Graph

from propnet.ext.matproj import MPRester

MPR = MPRester()
graph_evaluator = Graph(parallel=True, max_workers=4)


# explicitly making this an OrderedDict so we can go back from the
# display name to the symbol name
SCALAR_SYMBOLS = OrderedDict({k: v for k, v in sorted(Registry("symbols").items(),
                                                      key=lambda x: x[1].display_names[0])
                              if ((v.category == 'property' or v.category == 'condition')
                                  and v.shape == 1)})
ROW_IDX_TO_SYMBOL_NAME = [symbol for symbol in SCALAR_SYMBOLS.keys()]

DEFAULT_ROWS = [
    {
        'Property': symbol.display_names[0],
        'Editable Value': None
    }
    for symbol in SCALAR_SYMBOLS.values()
]


REMAINING_SYMBOLS = OrderedDict({k: v for k, v in sorted(Registry("symbols").items(),
                                                         key=lambda x: x[1].display_names[0])
                                if not ((v.category == 'property' or v.category == 'condition')
                                        and v.shape == 1)})
REMAINING_ROW_IDX_TO_SYMBOL_NAME = [symbol for symbol in REMAINING_SYMBOLS.keys()]

REMAINING_DEFAULT_ROWS = [
    {
        'Property': symbol.display_names[0],
        'Value': None
    }
    for symbol in REMAINING_SYMBOLS.values()
]


def interactive_layout(app):

    layout = html.Div([
        dcc.Markdown(
            'Generate materials properties using propnet in your browser, '
            'without needing to download any code.'),
        dcc.Markdown('## input data'),
        dcc.Markdown(
            'You can also pre-populate input data from the Materials Project '
            'database by entering a formula or Materials Project ID:'),
        html.Div([dcc.Input(
            placeholder='Enter a formula or mp-id...',
            type='text',
            value='',
            id='query-input',
            style={"width": "40%", "display": "inline-block", "vertical-align": "middle"}
        ),
        html.Button('Load data from Materials Project', id='submit-query',
            style={"display": "inline-block", "vertical-align": "middle"}),
            html.Button('Clear', id='clear-mp',
                        style={"display": "inline-block",
                               "vertical-align": "middle"})
        ]),
        html.Br(),
        html.Div(children=[dt.DataTable(id='mp-table',
                               rows=[{'Property': None, 'Materials Project Value': None}],
                               editable=False)], id='mp-container'),
        dcc.Store(id='mp-data', storage_type='memory'),
        html.Br(),
        dcc.Markdown(
            'You can also enter your own values of properties below. If units are not '
            'specified, default propnet units will be assigned, but you can '
            'also enter your own units.'),
        dt.DataTable(id='input-table', rows=DEFAULT_ROWS),
        html.Br(),
        dcc.Markdown('## propnet-derived output'),
        dcc.Markdown('Properties derived by propnet will be show below. If there are multiple '
                     'values for the same property, you can choose to aggregate them together.'
                     ''
                     ''
                     'In the graph, input properties are in green and derived properties in '
                     'yellow. Properties shown in grey require additional information to derive.'),
        dcc.Checklist(id='aggregate', options=[{'label': 'Aggregate', 'value': 'aggregate'}], values=['aggregate'], style={'display': 'inline-block'}),
        html.Br(),
        html.Div(id='propnet-output')
    ])

    @app.callback(Output('mp-data', 'data'),
                  [Input('submit-query', 'n_clicks'),
                   Input('query-input', 'n_submit')],
                  [State('query-input', 'value')])
    def retrieve_material(n_clicks, n_submit, query):

        if (n_clicks is None) and (n_submit is None):
            raise PreventUpdate

        if query.startswith("mp-") or query.startswith("mvc-"):
            mpid = query
        else:
            mpid = MPR.get_mpid_from_formula(query)

        material = MPR.get_material_for_mpid(mpid)
        if not material:
            raise PreventUpdate

        logger.info("Retrieved material {} for formula {}".format(
            mpid, material['pretty_formula']))

        mp_quantities = {quantity.symbol.display_names[0]: quantity.as_dict()
                         for quantity in material.get_quantities()}

        return json.dumps(mp_quantities, cls=MontyEncoder)

    @app.callback(
        Output('mp-container', 'style'),
        [Input('mp-data', 'data')]
    )
    def show_mp_table(data):
        if (data is None) or (len(data) == 0):
            return {'display': 'none'}
        else:
            return {}

    @app.callback(
        Output('mp-table', 'rows'),
        [Input('mp-data', 'data')]
    )
    def show_mp_table(data):
        if (data is None) or (len(data) == 0):
            raise PreventUpdate

        mp_quantities = json.loads(data, cls=MontyDecoder)

        output_rows = [
            {
                'Property': symbol_string,
                'Materials Project Value': quantity.pretty_string(sigfigs=3)
            }
            for symbol_string, quantity in mp_quantities.items()
        ]

        return output_rows


    @app.callback(Output('storage', 'clear_data'),
                  [Input('clear-mp', 'n_clicks')])
    def clear_data(n_clicks):
        if n_clicks is None:
            raise PreventUpdate
        return True

    @app.callback(
        Output('propnet-output', 'children'),
        [Input('input-table', 'rows'),
         Input('mp-data', 'data'),
         Input('aggregate', 'values')]
    )
    def evaluate(input_rows, data, aggregate):

        quantities = [QuantityFactory.create_quantity(symbol_type=ROW_IDX_TO_SYMBOL_NAME[idx],
                                                      value=ureg.parse_expression(row['Editable Value']))
                      for idx, row in enumerate(input_rows) if row['Editable Value']]

        if data and len(data) > 0:
            quantities += json.loads(data, cls=MontyDecoder).values()

        if not quantities:
            raise PreventUpdate

        material = Material()

        for quantity in quantities:
            material.add_quantity(quantity)

        output_material = graph_evaluator.evaluate(material)

        if aggregate:
            output_quantities = output_material.get_aggregated_quantities().values()
        else:
            output_quantities = output_material.get_quantities()

        output_rows = [{
            'Property': quantity.symbol.display_names[0],
            'Value': quantity.pretty_string(sigfigs=3)
        } for quantity in output_quantities]

        output_table = dt.DataTable(id='output-table',
                                    rows=output_rows,
                                    editable=False)

        # TODO: clean up

        input_quantity_names = [q.symbol.name for q in quantities]
        derived_quantity_names = set(
            [q.symbol.name for q in output_quantities]) - \
                                 set(input_quantity_names)
        material_graph_data = graph_conversion(
            graph_evaluator.get_networkx_graph(), nodes_to_highlight_green=input_quantity_names,
            nodes_to_highlight_yellow=list(derived_quantity_names))
        options = AESTHETICS['global_options']
        options['edges']['color'] = '#000000'
        output_graph = html.Div(GraphComponent(
            id='material-graph',
            graph=material_graph_data,
            options=options
        ), style={'width': '100%', 'height': '400px'})

        return [
            output_graph,
            html.Br(),
            output_table
        ]

    return layout
