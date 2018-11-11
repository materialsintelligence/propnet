import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt

from dash.dependencies import Input, Output, State

from propnet import log_stream
from propnet.web.layouts_models import model_layout, models_index
from propnet.web.layouts_symbols import symbol_layout, symbols_index
from propnet.web.layouts_plot import plot_layout
from propnet.web.layouts_home import home_layout
from propnet.web.layouts_interactive import interactive_layout

from mp_dash_components import GraphComponent
from propnet.web.utils import graph_conversion, parse_path, AESTHETICS
from propnet.core.graph import Graph

from propnet.ext.matproj import MPRester
from pydash import set_

from flask_caching import Cache
import logging

log = logging.getLogger(__name__)

# TODO: Fix math rendering

app = dash.Dash()
server = app.server
app.config.supress_callback_exceptions = True  # TODO: remove this?
app.scripts.config.serve_locally = True
app.title = "propnet"
route = dcc.Location(id='url', refresh=False)

cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem', 'CACHE_DIR': '.tmp'
})

mpr = MPRester()

g = Graph().get_networkx_graph()

# Define default graph component
@app.callback(Output('graph_explorer', 'children'),
              [Input('graph_options', 'values')])
def get_graph_component(props):
    aesthetics = AESTHETICS.copy()
    show_properties = 'show_properties' in props
    show_models = 'show_models' in props
    #print(props)
    #print("Updating graph to {}, {}".format(show_properties, show_models))
    set_(aesthetics, "node_aesthetics.Symbol.show_labels", show_properties)
    set_(aesthetics, "node_aesthetics.Model.show_labels", show_models)
    graph_data = graph_conversion(g, aesthetics=aesthetics)
    #print(graph_data)
    from uuid import uuid4
    graph_component = html.Div(
        id=str(uuid4()),
        children=[GraphComponent(id=str(uuid4()), graph=graph_data,
                                 options=AESTHETICS['global_options'])],
        style={'width': '100%', 'height': '800px'})
    return [graph_component]


# Do I really need to redo this code?
graph_data = graph_conversion(g)
graph_component = html.Div(
    id='graph_component',
    children=[GraphComponent(id='propnet-graph', graph=graph_data,
                             options=AESTHETICS['global_options'])],
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

layout_menu = html.Div(
    children=[dcc.Link('What is propnet?', href='/home'),
              html.Span(' • '),
              dcc.Link('Explore', href='/graph'),
              html.Span(' • '),
              dcc.Link('Models', href='/model'),
              html.Span(' • '),
              dcc.Link('Properties', href='/property'),
              html.Span(' • '),
              dcc.Link('Generate', href='/generate'),
              html.Span(' • '),
              dcc.Link('Plot', href='/plot')
              ])

# home page
home_manifesto = """
**Not intended for public use at this time.**
"""

home = html.Div([html.Br(),
                 dcc.Markdown(home_manifesto)])

# header
app.layout = html.Div(
    children=[route,
              html.Div([html.H3(app.title), layout_menu, html.Br()],
                       style={'textAlign': 'center'}),
              html.Div(id='page-content'),
              # hidden table to make sure table component loads
              # (Dash limitation; may be removed in future)
              html.Div(children=[dt.DataTable(rows=[{}]), graph_layout],
                       style={'display': 'none'})],
    style={'marginLeft': 200, 'marginRight': 200, 'marginTop': 30})

# standard Dash css, fork this for a custom theme
# we real web devs now
app.css.append_css(
    {'external_url': 'https://codepen.io/mkhorton-the-reactor/pen/oQbddV.css'})
# app.css.append_css(
#     {'external_url': 'https://codepen.io/montoyjh/pen/YjPKae.css'})
app.css.append_css(
    {'external_url': 'https://codepen.io/mikesmith1611/pen/QOKgpG.css'})

PLOT_LAYOUT = plot_layout(app)
INTERACTIVE_LAYOUT = interactive_layout(app)

# routing, current routes defined are:
# / for home page
# /model for model summary
# /model/model_name for information on that model
# /property for property summary
# /property/property_name for information on that property


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    """

    Args:
      pathname:

    Returns:

    """

    path_info = parse_path(pathname)

    if path_info:
        if path_info['mode'] == 'model':
            if path_info['value']:
                return model_layout(path_info['value'])
            else:
                return models_index
        elif path_info['mode'] == 'property':
            if path_info['value']:
                property_name = path_info['value']
                return symbol_layout(property_name)
            else:
                return symbols_index()
        elif path_info['mode'] == 'graph':
            return graph_layout
        elif path_info['mode'] == 'plot':
            return PLOT_LAYOUT
        elif path_info['mode'] == 'generate':
            return INTERACTIVE_LAYOUT
        elif path_info['mode'] == 'home':
            return home_layout()
        else:
            return '404'
    else:
        return home


if __name__ == '__main__':
    app.run_server(debug=True)
