import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from dash_cytoscape import Cytoscape
from propnet.web.utils import graph_conversion, STYLESHEET
from propnet.core.graph import Graph

from propnet.web.layouts_models import models_index
from propnet.web.layouts_symbols import symbols_index

from pydash import set_

from uuid import uuid4


def explore_layout(app):

    g = Graph().get_networkx_graph()

    graph_data = graph_conversion(g)
    graph_component = html.Div(
        id='graph_component',
        children=[Cytoscape(id='propnet-graph', elements=graph_data,
                            stylesheet=STYLESHEET,
                            layout={'name': 'cose',
                                    'animate': True,
                                    'animationThreshold': 10000,
                                    'randomize': True})],
        style={'width': '100%', 'height': '800px'})

    graph_layout = html.Div(
        id='graph_top_level',
        children=[
            dcc.Checklist(id='graph_options',
                          options=[{'label': 'Show models',
                                    'value': 'show_models'},
                                   {'label': 'Show properties',
                                    'value': 'show_properties'}],
                          values=['show_properties'],
                          labelStyle={'display': 'inline-block'}),
            html.Div(id='graph_explorer',
                     children=[graph_component])])

    # Define default graph component
    # TODO: this looks bad, re-evaluate
    '''
    @app.callback(Output('graph_explorer', 'children'),
                  [Input('graph_options', 'values')])
    def get_graph_component(props):
        
        aesthetics = AESTHETICS.copy()
        show_properties = 'show_properties' in props
        show_models = 'show_models' in props
        set_(aesthetics, "node_aesthetics.Symbol.show_labels", show_properties)
        set_(aesthetics, "node_aesthetics.Model.show_labels", show_models)
        graph_data = graph_conversion(g, aesthetics=aesthetics)
        graph_component = html.Div(
            id=str(uuid4()),
            children=[Cytoscape(id=str(uuid4()), elements=graph_data,
                                layout=AESTHETICS['global_options'])],
            style={'width': '100%', 'height': '800px'})
        return [graph_component]
    '''

    layout = html.Div([html.Div([graph_layout], className='row'),
                       html.Div([html.Div([models_index], className='six columns'),
                                 html.Div([symbols_index()], className='six columns'),], className='row')])

    return layout
