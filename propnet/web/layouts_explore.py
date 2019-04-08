import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from dash_cytoscape import Cytoscape
from propnet.web.utils import graph_conversion, STYLESHEET_FILE
from propnet.core.graph import Graph

from propnet.web.layouts_models import models_index
from propnet.web.layouts_symbols import symbols_index

from pydash import set_

from uuid import uuid4

from monty.serialization import loadfn
from os import path

GRAPH_LAYOUT_FILE = path.join(path.dirname(__file__), 'graph_layout.yaml')
propnet_nx_graph = Graph().get_networkx_graph()
graph_height_px = 1000


def explore_layout(app):
    graph_data = graph_conversion(propnet_nx_graph, graph_size_pixels=graph_height_px)
    layout = loadfn(GRAPH_LAYOUT_FILE)
    stylesheet = loadfn(STYLESHEET_FILE)
    graph_component = html.Div(
        id='graph_component',
        children=[Cytoscape(id='pn-graph', elements=graph_data,
                            style={'width': '100%',
                                   'height': str(graph_height_px) + "px"},
                            stylesheet=stylesheet,
                            layout=layout,
                            boxSelectionEnabled=True)],
        )

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

    @app.callback(Output('pn-graph', 'elements'),
                  [Input('graph_options', 'values')],
                  [State('pn-graph', 'elements')])
    def get_graph_component(props, elements):
        show_properties = 'show_properties' in props
        show_models = 'show_models' in props

        for elem in elements:
            group = elem['group']
            if group == 'edge':
                # applies to nodes only
                continue
            classes = elem.get('classes')
            if not classes:
                # if there is no classes specified, not sure what it is otherwise
                continue
            classes_list = classes.split(" ")
            is_model = any(c.startswith("model") for c in classes_list)
            is_symbol = any(c.startswith("symbol") for c in classes_list)

            if not is_model and not is_symbol:
                # is some other element on the graph, like the "unattached" model
                continue

            class_to_add = 'label-off'
            if (is_model and show_models) or (is_symbol and show_properties):
                class_to_add = 'label-on'

            for val in ('label-on', 'label-off'):
                try:
                    classes_list.remove(val)
                except ValueError:
                    pass
            classes_list.append(class_to_add)

            elem['classes'] = " ".join(classes_list)

        return elements

    layout = html.Div([html.Div([graph_layout], className='row'),
                       html.Div([html.Div([models_index], className='six columns'),
                                 html.Div([symbols_index()], className='six columns'),], className='row')])

    return layout
