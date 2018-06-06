from collections import OrderedDict

import dash_html_components as html
import dash_core_components as dcc
from dash_react_graph_vis import GraphComponent

import networkx as nx
from propnet.core.graph import Graph
from propnet.core.node import Node, NodeType

import propnet.models as models
from propnet.symbols import DEFAULT_SYMBOLS
from propnet.web.utils import references_to_markdown, graph_conversion, AESTHETICS

# layouts for model detail pages

def model_layout(model_name):
    """Create a Dash layout for a provided model.

    Args:
      model: an instance of an AbstractModel subclass

    Returns:
      Dash layout

    """

    # dict to hold layouts for each section
    layouts = OrderedDict()

    # instantiate model from name
    model = getattr(models, model_name)()

    model_title = html.Div(
        className='row',
        children=[
            html.H3(model.title),
        ]
    )

    # TODO: costly, should just construct subgraph directly?
    g = Graph()
    n = Node(node_type=NodeType.Model, node_value=model)
    subgraph = nx.ego_graph(g.graph, n, undirected=True)
    options=AESTHETICS['global_options']
    if "arrows" in options["edges"]:
        options["edges"]["arrows"] = "to"
    layouts['Graph'] = html.Div(
        GraphComponent(
            id="model_graph",
            graph=graph_conversion(subgraph),
            options=AESTHETICS['global_options']
        ),
        style={'width': '100%', 'height': '300px'}
    )

    if model.tags:
        tags = html.Ul(
            className="tags",
            children=[
                html.Li(tag, className="tag") for tag in model.tags
            ]
        )

        layouts['Tags'] = tags

    if model.references:
        references = html.Div([dcc.Markdown(references_to_markdown(ref))
                               for ref in model.references])

        layouts['References'] = references

    symbols = html.Div(
        children=[
            html.Div(
                className='row',
                children=[
                    html.Div(
                        className='two columns',
                        children=[
                            str(symbol)
                        ]
                    ),
                    html.Div(
                        className='ten columns',
                        children=[
                            dcc.Link(DEFAULT_SYMBOLS[symbol_name].display_names[0],
                                     href='/property/{}'.format(symbol_name))
                        ]
                    )
                ]
            )
            for symbol, symbol_name in model.symbol_mapping.items()
        ]
    )

    layouts['Symbols'] = symbols

    layouts['Description'] = dcc.Markdown(model.description)

    if model.test_data:
        sample_data_header = html.Div(
            className='row',
            children=[
                html.Div(
                    className='five columns',
                    style={'text-align': 'center'},
                    children=[
                        html.H4('Input(s)')
                    ]
                ),
                html.Div(
                    className='two columns',
                    style={'text-align': 'center'},
                    children=[
                        html.H4('->')
                    ]
                ),
                html.Div(
                    className='five columns',
                    style={'text-align': 'center'},
                    children=[
                        html.H4('Output(s)')
                    ]
                )
            ]
        )

        layouts['Sample Code'] = dcc.Markdown('```\n{}```'.format(model._example_code))

    sublayouts = []
    for title, layout in layouts.items():
        sublayouts.append(html.H6(title))
        sublayouts.append(layout)

    return html.Div([
        model_title,
        html.Br(),
        *sublayouts,
        html.Br(),
        dcc.Link('< Back to Models', href='/model'),
        html.Br(),
        dcc.Link('<< Back to Home', href='/')
    ])


model_links = {}
for model_name in models.DEFAULT_MODEL_NAMES:
    # instantiate model from name
    model = getattr(models, model_name)()

    # group by tag
    tags = model.tags

    for tag in tags:
        if tag not in model_links:
            model_links[tag] = []

    for tag in tags:
        passes = model.test()
        passes = "✅" if passes else "❌"

        link_text = "{}".format(model.title)

        model_links[tag].append(
            html.Div([
                html.Span('{} '.format(passes)),
                dcc.Link(link_text,
                         href='/model/{}'.format(model_name)),
                html.Br()
            ])
        )

model_links_grouped = []
for tag, links in model_links.items():
    model_links_grouped += [
        html.H6(tag.title()),
        html.Div(links),
        html.Br()
    ]

models_index = html.Div([
    html.H5('Current models:'),
    html.Div(model_links_grouped),
    html.Br(),
    dcc.Link('< Back', href='/')
])
