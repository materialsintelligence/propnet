import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt

from dash.dependencies import Input, Output, State

# these two imports may be removed in future
from flask import request
from urllib.parse import urlparse

from propnet import log_stream, ureg
from propnet.web.layouts_models import model_layout, models_index
from propnet.web.layouts_symbols import symbol_layout, symbols_index
from propnet.models import DEFAULT_MODEL_NAMES
from propnet.symbols import DEFAULT_SYMBOL_TYPE_NAMES

from force_graph import ForceGraphComponent
from propnet.web.utils import graph_conversion, parse_path
from propnet.core.graph import Propnet

from propnet.ext.matproj import PROPNET_PROPERTIES_ON_MP, MP_FROM_PROPNET_NAME_MAPPING,\
    materials_from_formula

from pymatgen import MPRester

import numpy as np
from scipy.stats import describe

from flask_caching import Cache

app = dash.Dash()
server = app.server
app.config.supress_callback_exceptions = True  # TODO: remove this?
app.scripts.config.serve_locally = True
app.title = ">>snappy title goes here<<"
route = dcc.Location(id='url', refresh=False)

cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem', 'CACHE_DIR': '.tmp'
})

mpr = MPRester()

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
    """

    Args:
      pathname: 

    Returns:

    """
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
    """

    Args:
      node: 

    Returns:

    """
    if not node:
        # This is a hack to get around a circular dependency
        # It is not nice!
        # It should be replaced.
        requesting_path = request.environ.get('HTTP_REFERER')
        return urlparse(requesting_path).path
    print(node)
    if node == 'home':
        return '/'
    elif node in DEFAULT_SYMBOL_TYPE_NAMES:
        return '/property/{}'.format(node)
    elif node in DEFAULT_MODEL_NAMES:
        return '/model/{}'.format(node)
    else:
        return '/'


layout_menu = html.Div(style={'textAlign': 'center'}, children=[
    dcc.Link('All Symbols', href='/property'),
    html.Span(' • '),
    dcc.Link('All Models', href='/model'),
    html.Span(' • '),
    dcc.Link('Explore with Materials Project', href='/load_material'),
])

# home page
index = html.Div([
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

We also provide interfaces to machine-learned models. Machine learning is **great**, and one
day might replace our conventional wisdom, but until then as scientists we still need to understand
how to use and interpret these machine-learned models. Additionally, formally codifying our
existing models will help train further machine-learned models in the future.
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
    html.H3(app.title),
    html.Br(),
    graph_component,
    html.Br(),
    layout_menu,
    html.Br(),
    html.Div(id='page-content'),
    dt.DataTable(rows=[{}])
], style={'marginLeft': 200, 'marginRight': 200, 'marginTop': 50})

# standard Dash css, fork this for a custom theme
app.css.append_css({
    'external_url': 'https://codepen.io/mkhorton/pen/zPgJJw.css'
})

# math rendering
# TODO: plot.ly uses MathJax too; we're probably loading this twice unnecessarily
app.scripts.append_script({
    'external_url': 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX'
                    '-MML-AM_CHTML'
})



@app.callback(
    Output('material-content', 'children'),
    [Input('submit-formula', 'n_clicks')],
    [State('formula-input', 'value')]
)
def retrieve_material(n_clicks, formula):

    materials = materials_from_formula(formula)

    if not materials:
        return "Material not found."

    material = materials[0]

    p = Propnet()
    g = p.graph

    available_properties = material.available_properties()

    rows = [
        {'property': str(property)} for property in available_properties
    ]

    table = dt.DataTable(
        rows=rows,
        row_selectable=True,
        filterable=True,
        sortable=True,
        selected_row_indices=[],
        id='datatable'
    )

    #p.evaluate(material)

    #derivable_properties = []
    #models_to_evaluate = []

    ## TODO: remove demo code
    ## this shouldn't be here
    ## it is also dumb dumb code
    ## and worse! it's wrong too, doesn't take into account routes
    #for node in p.graph:
    #    if not isinstance(node, Enum):
    #        in_edges = [e1 for e1, e2 in p.graph.in_edges(node)]
    #        in_edge_names = [in_edge.name for in_edge in in_edges]
    #        if all([prop_name in available_properties for prop_name in in_edge_names]):
    #            for e1, e2 in p.graph.out_edges(node):
    #                if isinstance(e2, Enum):
    #                    if e2.name not in available_properties:
    #                        models_to_evaluate.append(node.__name__)
    #                        models_to_evaluate.append(e2.name)
    #                        derivable_properties.append(e2.name)
#
#
    material_graph_data = graph_conversion(g, highlight=True,
                                           highlight_green=available_properties)
    material_graph_component = ForceGraphComponent(id='propnet-graph',
                                                   graphData=material_graph_data,
                                                   width=800,
                                                   height=350)

    return html.Div([
        html.Div(str(material)),
        #dcc.Link(mpid, href="https://materialsproject.org/materials/{}".format(mpid)),
        html.H3('Graph'),
        material_graph_component,
        html.H3('Table'),
        table,
        dcc.Markdown('### Available Properties\n\n{}'.format("\n\n".join(available_properties))),
    ])


material_layout = html.Div([
    dcc.Input(
        placeholder='Enter a formula...',
        type='text',
        value='',
        id='formula-input'
    ),
    html.Button('Load Material', id='submit-formula'),
    html.Button('Derive Properties', id='derive-properties'),
    html.Br(),
    html.Br(),
    html.Div(id='material-content')
])

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
        elif path_info['mode'] == 'load_material':
            return material_layout
        else:
            return '404'
    else:
        return index


if __name__ == '__main__':
    app.run_server(debug=False)
