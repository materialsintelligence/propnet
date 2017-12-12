import dash
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output

# these two imports may be removed in future
from flask import request
from urllib.parse import urlparse

from propnet import log_stream
from propnet.web.app_model import model_layout, models_index
from propnet.web.app_property import property_layout, properties_index
from propnet.models import all_model_names
from propnet.properties import all_property_names

from force_graph import ForceGraphComponent
from propnet.web.utils import graph_conversion, parse_path
from propnet.core.graph import Propnet

app = dash.Dash()
server = app.server
app.config.supress_callback_exceptions = True  # TODO: remove this?
app.scripts.config.serve_locally = True
app.title = "The Hitchhikers Guide to Materials Science"
route = dcc.Location(id='url', refresh=False)

g = Propnet().graph
graph_data = graph_conversion(g)
graph_component = html.Div(id='graph', children=[
    ForceGraphComponent(
        id='propnet-graph',
        graphData=graph_data,
        width=800,
        height=350
)], className='box')

# highlight node for corresponding content
@app.callback(
    Output('propnet-graph', 'selectedNode'),
    [Input('url', 'pathname')]
)
def hightlight_node_for_content(pathname):
    path_info = parse_path(pathname)
    if path_info and path_info['value']:
        return path_info['value']
    else:
        return 'none'

# display corresponding content when node clicked
@app.callback(
    Output('url', 'pathname'),
    [Input('propnet-graph', 'requestContent')]
)
def show_content_for_selected_node(node):
    if not node:
        # This is a hack to get around a circular dependency
        # It is not nice!
        # It should be replaced.
        requesting_path = request.environ.get('HTTP_REFERER')
        return urlparse(requesting_path).path
    print(node)
    if node == 'home':
        return '/'
    elif node in all_property_names:
        return '/property/{}'.format(node)
    elif node in all_model_names:
        return '/model/{}'.format(node)
    else:
        return '/'

# home page
index = html.Div([
    html.Div(style={'textAlign': 'center'}, children=[
        dcc.Link('All Properties', href='/property'),
        html.Span(' • '),
        dcc.Link('All Models', href='/model'),
        html.Span(' • '),
        dcc.Link('Load Material'),
        html.Span(' • '),
        dcc.Link('Developer Tools'),
        html.Span(' • '),
        dcc.Link('Fundamental Constants')
    ]),
    html.Br(),
    dcc.Markdown('''
**Under active development, pre-alpha.**
Real materials are complex. In the field of Materials Science, we often rely on empirical
relationships and rules-of-thumb to provide insights into the behavior of materials. This project
is designed to codify our knowledge of these empirical models and the relationships between them,
along with providing tested, unit-aware implementations of each model.

When given a set of known properties of a material, the knowledge graph can help
derive additional properties automatically. Integration with the
[Materials Project](https://materialsproject.org) and other databases
provides these sets of initial properties for a given material,
as well as information on the real world distributions of these properties.
    '''),
    dcc.Markdown('''
```
Propnet initialization log:
{}
```'''.format(log_stream.getvalue()))
])

# header
app.layout = html.Div(children=[
    route,
    html.H3("The Hitchhikers Guide to Materials Science "),
    html.Br(),
    graph_component,
    html.Br(),
    html.Div(id='page-content')
], style={'marginLeft': 200, 'marginRight': 200, 'marginTop': 50})

# standard Dash css, fork this for a custom theme
app.css.append_css({
    'external_url': 'https://codepen.io/mkhorton/pen/zPgJJw.css'
})

# math rendering
# TODO: plot.ly uses MathJax too; we're probably loading this twice unnecessarily
app.scripts.append_script({
    'external_url': 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML'
})

# routing, current routes defined are:
# / for home page
# /model for model summary
# /model/model_name for information on that model
# /property for property summary
# /property/property_name for information on that property
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):

    path_info = parse_path(pathname)

    if path_info:
        if path_info['mode'] == 'model':
            if path_info['value']:
                return model_layout(path_info['value'])
            else:
                return models_index
        elif path_info['mode'] == 'property':
            if path_info['value']:
                return property_layout(path_info['value'])
            else:
                return properties_index
        else:
            return '404'
    else:
        return index

if __name__ == '__main__':
    app.run_server(debug=False)