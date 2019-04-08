import dash_html_components as html
import dash_core_components as dcc
from dash_cytoscape import Cytoscape
from pydash import set_

import networkx as nx
from propnet.core.graph import Graph

from propnet.web.utils import graph_conversion, GRAPH_CONFIG, GRAPH_STYLESHEET

# noinspection PyUnresolvedReferences
import propnet.symbols
from propnet.core.registry import Registry

SUBGRAPH_SIZE_PX = 500
# layouts for symbol detail pages
def symbol_layout(symbol_name):
    """Create a Dash layout for a provided symbol.

    Args:
      symbol_name (str): a symbol name
      aesthetics (dict): an aesthetics configuration dictionary

    Returns:
      Dash layout

    """
    # aesthetics = aesthetics or AESTHETICS.copy()
    aesthetics = {}
    # list to hold layouts for each section
    layouts = []

    symbol = Registry("symbols")[symbol_name]

    main_name = symbol.display_names[0]

    layouts.append(html.H6('Graph'))
    # TODO: costly, should just construct subgraph directly?
    g = Graph()
    subgraph = nx.ego_graph(g.get_networkx_graph(), symbol, undirected=True, radius=2)
    subgraph_data = graph_conversion(subgraph, graph_size_pixels=SUBGRAPH_SIZE_PX,
                                     show_models=True, show_symbols=True)

    layouts.append(html.Div(
        Cytoscape(
            id="model_graph",
            elements=subgraph_data,
            style={'width': '100%',
                   'height': str(SUBGRAPH_SIZE_PX) + "px"},
            stylesheet=GRAPH_STYLESHEET,
            layout=GRAPH_CONFIG,
            boxSelectionEnabled=True
        )
    ))

    if len(symbol.display_names) > 1:
        display_names = ", ".join(symbol.display_names[1:])
        other_names = dcc.Markdown("Also known as: {}".format(display_names))
        layouts.append(other_names)

    if len(symbol.display_symbols) > 1:
        symbols = " ".join(symbol.display_symbols)
        symbols = dcc.Markdown("Common symbols: {}".format(symbols))
        layouts.append(symbols)

    if symbol.category in ('property', 'condition'):
        units = dcc.Markdown("Canonical units: **{}**".format(symbol.unit_as_string))
        dimension = dcc.Markdown("**{}**".format(symbol.dimension_as_string))
        layouts.append(units)
        layouts.append(dimension)

    if symbol.comment:
        layouts.append(dcc.Markdown(symbol.comment))

    return html.Div([
        main_name,
        html.Br(),
        html.Div(layouts),
        html.Br(),
        #dcc.Link('< Back to Properties', href='/property'),
        #html.Br(),
        dcc.Link('< Back', href='/explore')
    ])


def symbols_index():

    symbol_links = {}
    for symbol_name in Registry("symbols"):

        # group by tag
        symbol_type = Registry("symbols")[symbol_name].category
        display_name = Registry("symbols")[symbol_name].display_names[0]

        if symbol_type not in symbol_links:
            symbol_links[symbol_type] = []

        symbol_links[symbol_type].append(
            html.Div([
                dcc.Link("{}".format(display_name),
                         href='/property/{}'.format(symbol_name)),
                html.Br()
            ])
        )

    symbol_links_grouped = []
    for symbol_type, links in symbol_links.items():
        symbol_links_grouped += [
            html.H6(symbol_type.title()),
            html.Div(links),
            html.Br()
        ]

    return html.Div([
    html.H5('Currently supported symbols:'),
    html.Div(symbol_links_grouped),
    #html.Br(),
    #dcc.Link('< Back', href='/')
])
