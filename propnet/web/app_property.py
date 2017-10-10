import dash_html_components as html
import dash_core_components as dcc

from propnet.properties import all_property_names, PropertyType

def property_layout(property_name):
    property_metadata = PropertyType[property_name].value
    return html.Div([
        html.H3("{}".format(property_metadata.display_names[0])),
        dcc.Markdown("""
Common names: **{p.display_names}**

Common symbols: **{p.display_symbols}**

Units: **{p.units}**

Dimension: **{p.dimension}**

Canonical Propnet name: **{name}**

Comment: {p.comment}
        """.format(name=property_name, p=property_metadata)),
        html.Br(),
        dcc.Link('< Back to Properties', href='/property'),
        html.Br(),
        dcc.Link('<< Back to Propnet Home', href='/')
    ])

property_links = html.Div([
    html.Div([dcc.Link(PropertyType[property_name].value.display_names[0],
                       href='/property/{}'.format(property_name)),
             html.Br()])
    for property_name in all_property_names
])

list_properties = html.Div([
    html.H5('Currently supported properties:'),
    property_links,
    html.Br(),
    dcc.Link('< Back', href='/')
])