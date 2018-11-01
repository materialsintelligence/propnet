import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt

from dash.dependencies import Input, Output, State

from collections import OrderedDict

from propnet.symbols import DEFAULT_SYMBOLS

from propnet import ureg
from propnet.core.quantity import Quantity
from propnet.core.materials import Material
from propnet.core.graph import Graph


# explicitly making this an OrderedDict so we can go back from the
# display name to the symbol name
SCALAR_SYMBOLS = OrderedDict({k: v for k, v in sorted(DEFAULT_SYMBOLS.items(), key=lambda x: x[1].display_names[0])
                              if (v.category == 'property' and v.shape == 1)})
ROW_IDX_TO_SYMBOL_NAME = [symbol for symbol in SCALAR_SYMBOLS.keys()]

DEFAULT_ROWS = [
    {
        'Property': symbol.display_names[0],
        'Value': None
    }
    for symbol in SCALAR_SYMBOLS.values()
]

def interactive_layout(app):

    layout = html.Div([
        dcc.Markdown('## Input'),
        dcc.Markdown(
            'Enter your own values of properties below. If units are not '
            'specified, default propnet units will be assigned, but you can '
            'also enter your own units. Derived quantities will be evaluated '
            'below.'),
        dt.DataTable(id='input-table', rows=DEFAULT_ROWS),
        dcc.Markdown('## Output'),
        html.Div(id='error'),
        dt.DataTable(id='output-table',
                     rows=[{'Property': None, 'Value': None, 'Provenance': None}],
                     editable=False)
    ])

    @app.callback(
        Output('output-table', 'rows'),
        [Input('input-table', 'rows')]
    )
    def evaluate(input_rows):

        quantities = [Quantity(symbol_type=ROW_IDX_TO_SYMBOL_NAME[idx],
                               value=ureg.parse_expression(row['Value']))
                      for idx, row in enumerate(input_rows) if row['Value']]
        material = Material()

        for quantity in quantities:
            material.add_quantity(quantity)

        graph = Graph()
        output_material = graph.evaluate(material)

        output_quantities = output_material.get_aggregated_quantities()
        print(output_quantities)

        output_rows = [{
            'Property': symbol.display_names[0],
            'Value': str(quantity.value),
            'Provenance': None
        } for symbol, quantity in output_quantities.items()]

        return output_rows

    return layout
