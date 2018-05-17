import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go

from propnet.symbols import DEFAULT_SYMBOLS


# layouts for symbol detail pages

def symbol_layout(symbol_name):

    symbol = DEFAULT_SYMBOLS[symbol_name]

    main_name = symbol.display_names[0]

    # TODO: clean up this layout (will probably be easier if symbol is refactored)

    if len(symbol.display_names) > 1:
        display_names = ", ".join(symbol.display_names[1:])
        other_names = dcc.Markdown("Also known as: {}".format(display_names))
    else:
        other_names = html.Div()

    if len(symbol.display_symbols) > 1:
        symbols = " ".join(symbol.display_symbols)
        symbols = dcc.Markdown("Common symbols: {}".format(symbols))
    else:
        symbols = html.Div()

    if symbol.category in ('property', 'condition'):
        units = dcc.Markdown("Canonical units: **{}**".format(symbol.unit_as_string))
        if symbol.compatible_units:
            compatible_units = dcc.Markdown('Compatible units:\n\n{}'
                                            .format('\n\n'.join(map("* {}".format,
                                                                    symbol.compatible_units))))
        else:
            compatible_units = html.Div()
        dimension = dcc.Markdown("**{}**".format(symbol.dimension_as_string))
    else:
        units = html.Div()
        compatible_units = html.Div()
        dimension = html.Div()

    if symbol.comment:
        comment = dcc.Markdown(symbol.comment)
    else:
        comment = html.Div()

    return html.Div([
        html.H3(main_name),
        other_names,
        symbols,
        dimension,
        comment,
        units,
        compatible_units,
        html.Br(),
        dcc.Link('< Back to Properties', href='/property'),
        html.Br(),
        dcc.Link('<< Back to Propnet Home', href='/')
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
