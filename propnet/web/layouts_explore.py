import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output, State

from dash_cytoscape import Cytoscape
from propnet.web.utils import graph_conversion, GRAPH_STYLESHEET, \
    GRAPH_LAYOUT_CONFIG, GRAPH_SETTINGS, propnet_nx_graph, update_labels


from propnet.web.layouts_models import models_index
from propnet.web.layouts_symbols import symbols_index


def explore_layout(app):
    graph_data = graph_conversion(propnet_nx_graph, hide_unconnected_nodes=False)
    graph_component = html.Div(
        id='graph_component',
        children=[Cytoscape(id='pn-graph', elements=graph_data,
                            stylesheet=GRAPH_STYLESHEET,
                            layout=GRAPH_LAYOUT_CONFIG,
                            **GRAPH_SETTINGS['full_view'])],
        )

    graph_layout = html.Div(
        id='graph_top_level',
        children=[
            dcc.Checklist(id='graph_options',
                          options=[{'label': 'Show models',
                                    'value': 'show_models'},
                                   {'label': 'Show properties',
                                    'value': 'show_properties'}],
                          value=['show_properties'],
                          labelStyle={'display': 'inline-block'}),
            html.Div(id='graph_explorer',
                     children=[graph_component])])

    @app.callback(Output('pn-graph', 'elements'),
                  [Input('graph_options', 'value')],
                  [State('pn-graph', 'elements')])
    def change_propnet_graph_label_selection(props, elements):
        show_properties = 'show_properties' in props
        show_models = 'show_models' in props

        update_labels(elements, show_models=show_models, show_symbols=show_properties)

        return elements

    layout = html.Div([html.Div([graph_layout], className='row'),
                       html.Div([html.Div([models_index], className='six columns'),
                                 html.Div([symbols_index()], className='six columns'),], className='row')])

    return layout
