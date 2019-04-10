from collections import OrderedDict

import dash_html_components as html
import dash_core_components as dcc
from dash_cytoscape import Cytoscape

import networkx as nx
from propnet.core.graph import Graph

from propnet.web.utils import graph_conversion, GRAPH_LAYOUT_CONFIG, \
    GRAPH_STYLESHEET, GRAPH_SETTINGS, propnet_nx_graph
from propnet.core.utils import references_to_markdown

# noinspection PyUnresolvedReferences
import propnet.symbols
# noinspection PyUnresolvedReferences
import propnet.models
from propnet.core.registry import Registry

import logging

logger = logging.getLogger(__name__)


# layouts for model detail pages
def model_layout(model_name):
    """Create a Dash layout for a provided model.

    Args:
      model_name: an instance of an AbstractModel subclass

    Returns:
      Dash layout

    """

    # dict to hold layouts for each section
    layouts = OrderedDict()

    # instantiate model from name
    model = Registry("models")[model_name]

    model_title = html.Div(
        className='row',
        children=[
            html.H3(model.title),
        ]
    )

    # TODO: costly, should just construct subgraph directly?

    subgraph = nx.ego_graph(propnet_nx_graph, model, undirected=True)
    subgraph_data = graph_conversion(subgraph,
                                     show_symbol_labels=True, show_model_labels=True)
    if len(subgraph_data) < 50:
        graph_config = GRAPH_LAYOUT_CONFIG.copy()
        graph_config['maxSimulationTime'] = 1500
    else:
        graph_config = GRAPH_LAYOUT_CONFIG

    layouts['Graph'] = html.Div(
        Cytoscape(
            id="model_graph",
            elements=subgraph_data,
            stylesheet=GRAPH_STYLESHEET,
            layout=graph_config,
            **GRAPH_SETTINGS['model_symbol_view']
        )
    )

    if model.categories:
        tags = html.Ul(
            className="tags",
            children=[html.Li(tag_, className="tag")
                      for tag_ in model.categories]
        )
        layouts['Tags'] = tags

    if model.references:
        markdown = []
        for ref in model.references:
            try:
                markdown.append(references_to_markdown(ref))
            except ValueError as ex:
                logger.error("Error with reference:\n{}\nReference:\n{}".format(ex, ref))
        references = html.Div([dcc.Markdown(ref)
                               for ref in markdown])

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
                            dcc.Link(Registry("symbols")[prop_name].display_names[0],
                                     href='/property/{}'.format(prop_name))
                        ]
                    )
                ]
            )
            for symbol, prop_name in model.symbol_property_map.items()
        ]
    )

    layouts['Symbols'] = symbols

    layouts['Description'] = dcc.Markdown(model.description)

    if model.validate_from_preset_test():
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

        layouts['Sample Code'] = html.Div([
            dcc.Markdown("Propnet models can be called directly, with propnet acting "
                         "as a library of tested materials science models. Sample code for this "
                         "model is as follows:"),
            dcc.SyntaxHighlighter(model.example_code)
        ])

    sublayouts = []
    for title, layout in layouts.items():
        sublayouts.append(html.H6(title))
        sublayouts.append(layout)

    return html.Div([
        model_title,
        html.Br(),
        *sublayouts,
        html.Br(),
        #dcc.Link('< Back to Models', href='/model'),
        #html.Br(),
        dcc.Link('< Back', href='/explore')
    ])


model_links = {}
for model_name, model in Registry("models").items():
    # group by tag
    for tag in model.categories:
        if tag not in model_links:
            model_links[tag] = []

    for tag in model.categories:
        passes = model.validate_from_preset_test()
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
    #html.Br(),
    #dcc.Link('< Back', href='/')
])
