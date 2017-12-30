import dash
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output, State

# these two imports may be removed in future
from flask import request
from urllib.parse import urlparse

from propnet import log_stream, ureg
from propnet.web.layouts_models import model_layout, models_index
from propnet.web.layouts_properties import property_layout, properties_index
from propnet.models import all_model_names
from propnet.vars import all_property_names

from force_graph import ForceGraphComponent
from propnet.web.utils import graph_conversion, parse_path
from propnet.core.graph import Propnet

from propnet.ext.matproj import mpr, PROPNET_PROPERTIES_ON_MP, MP_FROM_PROPNET_NAME_MAPPING, \
    mpid_from_formula, materials_from_mp_ids
import numpy as np
from scipy.stats import describe

from flask_caching import Cache

app = dash.Dash()
server = app.server
app.config.supress_callback_exceptions = True  # TODO: remove this?
app.scripts.config.serve_locally = True
app.title = "The Hitchhikers Guide to Materials Science"
route = dcc.Location(id='url', refresh=False)

cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem', 'CACHE_DIR': 'tmp'
})


@cache.memoize(timeout=86400)
def retrieve_mp_properties(property_name):
    # this horrific function *will* be replaced, for demo only
    try:
        mp_property = MP_FROM_PROPNET_NAME_MAPPING[property_name]
        query_result = mpr.query({}, properties=['task_id', mp_property])
    except:
        return (None, None)
    try:
        list_of_property_values = [r[mp_property] for r in query_result]
        hist = np.histogram(list_of_property_values, bins='auto')
        description = describe(list_of_property_values)
    except:
        return (None, None)
    return (hist, description)


available_hists = {name: False for name in all_property_names}
available_hists = {name: bool(retrieve_mp_properties(name)[0]) for name in all_property_names}

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
    elif node in all_property_names:
        return '/property/{}'.format(node)
    elif node in all_model_names:
        return '/model/{}'.format(node)
    else:
        return '/'


layout_menu = html.Div(style={'textAlign': 'center'}, children=[
    dcc.Link('All Properties', href='/property'),
    html.Span(' • '),
    dcc.Link('All Models', href='/model'),
    html.Span(' • '),
    dcc.Link('Load Material', href='/load_material'),
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
    html.H3("The Hitchhikers Guide to Materials Science "),
    html.Br(),
    graph_component,
    html.Br(),
    layout_menu,
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
    'external_url': 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX'
                    '-MML-AM_CHTML'
})



from enum import Enum
@app.callback(
    Output('material-content', 'children'),
    [Input('submit-formula', 'n_clicks')],
    [State('formula-input', 'value')]
)
def retrieve_material(n_clicks, formula):
    mpid = mpid_from_formula(formula)

    material = materials_from_mp_ids([mpid])[0]

    p = Propnet()
    g = p.graph

    available_properties = material.available_properties()

    derivable_properties = []
    models_to_evaluate = []

    # TODO: remove demo code
    # this shouldn't be here
    # it is also dumb dumb code
    # and worse! it's wrong too, doesn't take into account routes
    for node in p.graph:
        if not isinstance(node, Enum):
            in_edges = [e1 for e1, e2 in p.graph.in_edges(node)]
            in_edge_names = [in_edge.name for in_edge in in_edges]
            if all([prop_name in available_properties for prop_name in in_edge_names]):
                for e1, e2 in p.graph.out_edges(node):
                    if isinstance(e2, Enum):
                        if e2.name not in available_properties:
                            models_to_evaluate.append(node.__name__)
                            models_to_evaluate.append(e2.name)
                            derivable_properties.append(e2.name)


    material_graph_data = graph_conversion(g, highlight=True,
                                           highlight_green=available_properties,
                                           highlight_yellow=models_to_evaluate)
    material_graph_component = ForceGraphComponent(id='propnet-graph',
                                                   graphData=material_graph_data,
                                                   width=800,
                                                   height=350)

    if mpid:
        return html.Div([
            dcc.Link(mpid, href="https://materialsproject.org/materials/{}".format(mpid)),
            material_graph_component,
            html.Br(),
            dcc.Markdown('### Available Properties\n\n{}'.format("\n\n".join(available_properties))),
            html.Br(),
            dcc.Markdown('### Derivable Properties\n\n{}'.format("\n\n".join(derivable_properties))),
        ])
    else:
        return None


#calculator_layouts = {}
#for property_name in all_property_names:
#    calculator = html.Div([
#        dcc.Input(
#            placeholder='Enter value',
#            type='number',
#            value='',
#            id='calculator-value-input-{}'.format(property_name)
#        ),
#        dcc.Input(
#            placeholder='Enter input unit',
#            type='text',
#            value='',
#            id='calculator-unit-input-{}'.format(property_name)
#        ),
#        dcc.Input(
#            placeholder='Enter desired output unit',
#            type='text',
#            value='',
#            id='calculator-unit-output-{}'.format(property_name)
#        ),
#        html.Button('Convert', id='calculator-button-{}'.format(property_name))
#        html.Br(),
#        html.Label('', id='calculator-output-{}'.format(property_name))
#    ])
#    calculator_layouts[property_name] = calculator
#
#    @app.callback(
#        Output('calculator-output-{}'.format(property_name), 'children'),
#        [Input('calculator-button-{}'.format(property_name), 'n_clicks')],
#        [State('calculator-value-input-{}'.format(property_name), 'value'),
#         State('calculator-unit-input-{}'.format(property_name), 'value'),
#         State('calculator-unit-output-{}'.format(property_name), 'value')]
#    )
#    def convert_units(n_clicks, value, input_unit, output_unit):
#        # TODO: remove try/except, add warning label div
#        try:
#            q = ureg.Quantity()
#        except:
#            return "Could not convert."#

material_layout = html.Div([
    dcc.Input(
        placeholder='Enter a formula...',
        type='text',
        value='',
        id='formula-input'
    ),
    html.Button('Submit', id='submit-formula'),
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
                if property_name in PROPNET_PROPERTIES_ON_MP:
                    mp_values = retrieve_mp_properties(property_name)
                    print(mp_values)
                else:
                    mp_values = None
                return property_layout(property_name, mp_values=mp_values)
            else:
                return properties_index(available_hists)
        elif path_info['mode'] == 'load_material':
            return material_layout
        else:
            return '404'
    else:
        return index


if __name__ == '__main__':
    app.run_server(debug=False)
