import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go

from propnet.symbols import all_property_names, PropertyType


# layouts for property detail pages

def property_layout(property_name, mp_values=None):
    """

    Args:
      property_name:

    Returns:

    """

    property_metadata = PropertyType[property_name].value

    main_name = property_metadata.display_names[0]

    if len(property_metadata.display_names) > 1:
        display_names = ", ".join(property_metadata.display_names[1:])
        other_names = dcc.Markdown("Also known as: {}".format(display_names))
    else:
        other_names = html.Div()

    if len(property_metadata.display_symbols) > 1:
        symbols = " ".join(property_metadata.display_symbols)
        symbols = dcc.Markdown("Common symbols: {}".format(symbols))
    else:
        symbols = html.Div()

    if property_metadata.type in ('property', 'condition'):
        units = dcc.Markdown("Units: **{}**".format(property_metadata.unit_as_string))
        dimension = dcc.Markdown("Type: **{}**".format(property_metadata.dimension_as_string))
    else:
        units = html.Div()
        dimension = html.Div()

    if property_metadata.comment:
        comment = dcc.Markdown(property_metadata.comment)
    else:
        comment = html.Div()

    if mp_values:

        hist = mp_values[0]
        if hist:
            graph = dcc.Graph(
                figure=go.Figure(
                    data=[
                        go.Bar(
                            x=hist[0],
                            y=hist[1],
                            name='Materials Project',
                            marker=go.Marker(
                                color='rgb(12, 156, 156)'
                            )
                        )
                    ],
                    layout=go.Layout(
                        title='Distribution of {} in available data'.format(main_name.lower()),
                        showlegend=True,
                        legend=go.Legend(
                            x=0,
                            y=1.0
                        ),
                        margin=go.Margin(l=40, r=0, t=40, b=30)
                    )
                ),
                style={'height': 300},
                id='property-graph'
            )
        else:
            graph = html.Div()

        mp = html.Div(['This property is available on the Materials Project.',
                       html.Br(),
                       html.Br(),
                       graph])
    else:
        mp = html.Div()

    return html.Div([
        html.H3(main_name),
        other_names,
        symbols,
        units,
        dimension,
        comment,
        html.Br(),
        mp,
        html.Br(),
        html.H5('Calculator'),
        html.Div(id='{}-calculator'.format(property_name)),
        dcc.Link('< Back to Properties', href='/property'),
        html.Br(),
        dcc.Link('<< Back to Propnet Home', href='/')
    ])


def properties_index(available_hists=None):

    # properties for which we have values from a database, e.g. MP
    # this was used in a demo, needs to be replaced with something
    # more permanent
    if not available_hists:
        available_hists = {}

    property_links = {}
    for property_name in all_property_names:

        # group by tag
        property_type = PropertyType[property_name].value.type
        # TODO: rename .type, add .display_name property, rename PropertyType etc.
        display_name = PropertyType[property_name].value.display_names[0]

        if property_type not in property_links:
            property_links[property_type] = []

        if available_hists.get(property_name, None):
            token = "• "
        else:
            token = ""

        property_links[property_type].append(
            html.Div([
                dcc.Link("{}{}".format(token, display_name),
                         href='/property/{}'.format(property_name)),
                html.Br()
            ])
        )

    property_links_grouped = []
    for property_type, links in property_links.items():
        property_links_grouped += [
            html.H6(property_type.title()),
            html.Div(links),
            html.Br()
        ]

    return html.Div([
    html.H5('Currently supported properties:'),
    html.Div(property_links_grouped),
    html.Br(),
    dcc.Link('< Back', href='/')
])
