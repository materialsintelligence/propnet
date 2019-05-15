import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt

from dash_cytoscape import Cytoscape
from propnet.web.utils import graph_conversion, GRAPH_LAYOUT_CONFIG, \
    GRAPH_SETTINGS, GRAPH_STYLESHEET, propnet_nx_graph, update_labels

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
from propnet.ext.aflow import AflowAdapter

MPR = MPRester()
AFA = AflowAdapter()
graph_evaluator = Graph(parallel=True, max_workers=4)

# explicitly making this an OrderedDict so we can go back from the
# display name to the symbol name
# Removed condition symbols from table until we can handle combinatorics blow-up that results
# from adding a temperature -cml
# TODO: Add condition symbols back when combinartorics problem solved
SCALAR_SYMBOLS = OrderedDict({k: v for k, v in sorted(Registry("symbols").items(),
                                                      key=lambda x: x[1].display_names[0])
                              if (v.category == 'property'
                                  and v.shape == 1)})
ROW_IDX_TO_SYMBOL_NAME = [symbol for symbol in SCALAR_SYMBOLS.keys()]

DEFAULT_ROWS = [
    {
        'Property': symbol.display_names[0],
        'Editable Value': ""
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
        'Value': ""
    }
    for symbol in REMAINING_SYMBOLS.values()
]

# I'm sure that the font stuff can be handled by css, but I can't css as of now...
DATA_TABLE_STYLE = dict(
    n_fixed_rows=1,
    style_table={
        'maxHeight': '400px',
        'overflowY': 'scroll'
    },
    style_cell={
        'minWidth': '400px', 'width': '400px', 'maxWidth': '400px',
        'whiteSpace': 'no-wrap',
        'overflow': 'hidden',
        'textOverflow': 'clip',
        'font-family': 'HelveticaNeue',
        'text-align': 'left'
    },
    style_data={
        'whiteSpace': 'normal'
    },
    style_header={
        'fontWeight': 'bold',
        'font-family': 'HelveticaNeue',
        'text-align': 'left'
    }
)


def interactive_layout(app):

    layout = html.Div([
        dcc.Markdown(
            'Generate materials properties using propnet in your browser, '
            'without needing to download any code.'),
        dcc.Markdown('## input data'),
        dcc.Markdown(
            'You can also pre-populate input data from the Materials Project '
            'database by entering a formula or Materials Project ID or with '
            'data from the AFLOW database by entering an AFLOW ID:'),
        html.Div([dcc.Input(
            placeholder='Enter a formula, mp-id, or auid...',
            type='text',
            value='',
            id='query-input',
            style={"width": "40%", "display": "inline-block", "vertical-align": "middle"}
        ),
        html.Button('Load external data', id='submit-query',
            style={"display": "inline-block", "vertical-align": "middle"}),
            html.Button('Clear', id='clear-db',
                        style={"display": "inline-block",
                               "vertical-align": "middle"})
        ]),
        html.Br(),
        dcc.Markdown(id='calculation-status'),
        html.Br(),
        html.Div(children=[dt.DataTable(id='db-table',
                                        data=[{'Property': "", 'Database Value': ""}],
                                        columns=[{'id': val, 'name': val}
                                                 for val in ('Property', 'Database Value')],
                                        editable=False, **DATA_TABLE_STYLE)], id='db-container'),
        dcc.Store(id='db-data', storage_type='memory'),
        html.Br(),
        dcc.Markdown(
            'You can also enter your own values of properties below. If units are not '
            'specified, default propnet units will be assigned, but you can '
            'also enter your own units.'),
        dt.DataTable(id='input-table', data=DEFAULT_ROWS,
                     columns=[{'id': val, 'name': val}
                              for val in ('Property', 'Editable Value')],
                     editable=True, **DATA_TABLE_STYLE
        ),
        html.Br(),
        dcc.Markdown('## propnet-derived output'),
        dcc.Markdown('Properties derived by propnet will be show below. If there are multiple '
                     'values for the same (scalar) property, you can choose to aggregate them together.'),
        dcc.Markdown('In the graph, input properties are in green and derived properties in '
                     'yellow. Models used to derive these properties are in blue. Properties '
                     'shown in grey require additional information to derive.'),
        dcc.Checklist(id='aggregate', options=[{'label': 'Aggregate', 'value': 'aggregate'}],
                      values=['aggregate'], style={'display': 'inline-block'}),
        html.Br(),
        html.Div(id='propnet-output'),
        html.Br()
    ])

    @app.callback(Output('db-data', 'data'),
                  [Input('submit-query', 'n_clicks'),
                   Input('query-input', 'n_submit')],
                  [State('query-input', 'value')])
    def retrieve_material(n_clicks, n_submit, query):

        if (n_clicks is None) and (n_submit is None) or query == "":
            raise PreventUpdate

        if query.startswith("aflow"):
            identifier = query
            material = AFA.get_material_by_auid(query)
            formula = material['formula']
        else:
            if query.startswith("mp-") or query.startswith("mvc-"):
                identifier = query
            else:
                identifier = MPR.get_mpid_from_formula(query)
            material = MPR.get_material_for_mpid(identifier)
            formula = material['pretty_formula']

        if not material:
            raise PreventUpdate

        logger.info("Retrieved material {} for formula {}".format(
            identifier, formula))

        db_quantities = {quantity.symbol.display_names[0]: quantity.as_dict()
                         for quantity in material.get_quantities()}

        return json.dumps(db_quantities, cls=MontyEncoder)

    @app.callback(
        Output('db-table', 'data'),
        [Input('db-data', 'data')]
    )
    def show_db_table(data):
        if (data is None) or (len(data) == 0):
            raise PreventUpdate

        db_quantities = json.loads(data, cls=MontyDecoder)

        output_rows = [
            {
                'Property': symbol_string,
                'Database Value': quantity.pretty_string(sigfigs=3)
            }
            for symbol_string, quantity in db_quantities.items()
        ]

        return output_rows

    @app.callback(Output('storage', 'clear_data'),
                  [Input('clear-db', 'n_clicks')])
    def clear_data(n_clicks):
        if n_clicks is None:
            raise PreventUpdate
        return True

    @app.callback(
        Output('propnet-output', 'children'),
        [Input('input-table', 'data'),
         Input('db-data', 'data'),
         Input('aggregate', 'values')]
    )
    def evaluate(input_rows, data, aggregate):

        quantities = [QuantityFactory.create_quantity(symbol_type=ROW_IDX_TO_SYMBOL_NAME[idx],
                                                      value=ureg.parse_expression(row['Editable Value']),
                                                      units=Registry("units").get(ROW_IDX_TO_SYMBOL_NAME[idx]))
                      for idx, row in enumerate(input_rows) if row['Editable Value']]

        if data and len(data) > 0:
            quantities += json.loads(data, cls=MontyDecoder).values()

        if not quantities:
            raise PreventUpdate

        material = Material()

        for quantity in quantities:
            material.add_quantity(quantity)

        output_material = graph_evaluator.evaluate(material, timeout=5)

        if aggregate:
            aggregated_quantities = output_material.get_aggregated_quantities()
            non_aggregatable_quantities = [v for v in output_material.get_quantities()
                                           if v.symbol not in aggregated_quantities]
            output_quantities = list(aggregated_quantities.values()) + non_aggregatable_quantities
        else:
            output_quantities = output_material.get_quantities()

        output_rows = [{
            'Property': quantity.symbol.display_names[0],
            'Value': quantity.pretty_string(sigfigs=3)
        } for quantity in output_quantities]

        output_table = dt.DataTable(id='output-table',
                                    data=output_rows,
                                    columns=[{'id': val, 'name': val}
                                             for val in ('Property', 'Value')],
                                    editable=False, **DATA_TABLE_STYLE)

        # TODO: clean up

        input_quantity_names = [q.symbol for q in quantities]
        derived_quantity_names = \
            set([q.symbol for q in output_quantities]) - \
            set(input_quantity_names)

        models_evaluated = set(output_q.provenance.model
                               for output_q in output_material.get_quantities())
        models_evaluated = [Registry("models").get(m) for m in models_evaluated
                            if Registry("models").get(m) is not None]

        material_graph_data = graph_conversion(
            propnet_nx_graph,
            derivation_pathway={'inputs': input_quantity_names,
                                'outputs': list(derived_quantity_names),
                                'models': models_evaluated})

        output_graph = html.Div(
            children=[
                dcc.Checklist(id='material-graph-options',
                              options=[{'label': 'Show models',
                                        'value': 'show_models'},
                                       {'label': 'Show properties',
                                        'value': 'show_properties'}],
                              values=['show_properties'],
                              labelStyle={'display': 'inline-block'}),
                Cytoscape(
                    id='material-graph',
                    elements=material_graph_data,
                    stylesheet=GRAPH_STYLESHEET,
                    layout=GRAPH_LAYOUT_CONFIG,
                    **GRAPH_SETTINGS['full_view']
                    )
            ]
        )

        return [
            output_graph,
            html.Br(),
            output_table
        ]

    @app.callback(Output('material-graph', 'elements'),
                  [Input('material-graph-options', 'values')],
                  [State('material-graph', 'elements')])
    def change_material_graph_label_selection(props, elements):
        show_properties = 'show_properties' in props
        show_models = 'show_models' in props

        update_labels(elements, show_models=show_models, show_symbols=show_properties)

        return elements

    return layout
