import dash_html_components as html
import dash_core_components as dcc
from mp_dash_components import GraphComponent
from pydash import set_

import networkx as nx
from propnet.core.graph import Graph

from propnet.symbols import DEFAULT_SYMBOLS
from propnet.web.utils import graph_conversion, AESTHETICS


# layouts for symbol detail pages
def symbol_layout(symbol_name, aesthetics=None):
    """Create a Dash layout for a provided symbol.

    Args:
      symbol_name (str): a symbol name
      aesthetics (dict): an aesthetics configuration dictionary

    Returns:
      Dash layout

    """
    aesthetics = aesthetics or AESTHETICS.copy()

    # list to hold layouts for each section
    layouts = []

    symbol = DEFAULT_SYMBOLS[symbol_name]

    main_name = symbol.display_names[0]

    layouts.append(html.H6('Graph'))
    # TODO: costly, should just construct subgraph directly?
    g = Graph()
    subgraph = nx.ego_graph(g.graph, symbol, undirected=True, radius=2)
    options=AESTHETICS['global_options']
    if "arrows" in options["edges"]:
        options["edges"]["arrows"] = "to"
    set_(aesthetics, "node_options.show_model_labels", True)
    layouts.append(html.Div(
        GraphComponent(
            id="model_graph",
            graph=graph_conversion(subgraph, aesthetics=AESTHETICS),
            options=AESTHETICS['global_options']
        ),
        style={'width': '100%', 'height': '300px'}
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
        dcc.Link('< Back to Properties', href='/property'),
        html.Br(),
        dcc.Link('<< Back to Home', href='/')
    ])


def symbols_index(available_hists=None):

    # properties for which we have values from a database, e.g. MP
    # this was used in a demo, needs to be replaced with something
    # more permanent
    if not available_hists:
        available_hists = {}

    symbol_links = {}
    for symbol_name in DEFAULT_SYMBOLS:

        # group by tag
        symbol_type = DEFAULT_SYMBOLS[symbol_name].category
        display_name = DEFAULT_SYMBOLS[symbol_name].display_names[0]

        if symbol_type not in symbol_links:
            symbol_links[symbol_type] = []

        if available_hists.get(symbol_name, None):
            token = "• "
        else:
            token = ""

        symbol_links[symbol_type].append(
            html.Div([
                dcc.Link("{}{}".format(token, display_name),
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
    html.Br(),
    dcc.Link('< Back', href='/')
])
