import dash_html_components as html
import dash_core_components as dcc

from propnet.properties import all_property_names, PropertyType

from pint import formatter
#formatter(unit.dimensionality)

# layouts for property detail pages

def property_layout(property_name):

    property_metadata = PropertyType[property_name].value

    main_name = property_metadata.display_names[0]

    if len(property_metadata.display_names) > 1:
        other_names = dcc.Markdown("Also known as: {}".format(property_metadata.display_names[1:]))
    else:
        other_names = None

    #units = dcc.Markdown("Units: ".format(property_metadata.))

    return html.Div([
        html.H3(main_name),
        other_names,

        dcc.Markdown("""
Common names: **{p.display_names}**

Common symbols: **{p.display_symbols}**

Units: **{p.units}**

Dimension: **{p.dimension}**

{p.comment}
        """.format(name=property_name, p=property_metadata)),
        html.Br(),
        dcc.Link('< Back to Properties', href='/property'),
        html.Br(),
        dcc.Link('<< Back to Propnet Home', href='/')
    ])


# need a 'warning' div for wrong units
calculator_layout = html.Div([
    html.H4('Calculator')
])

list_of_properties = html.Div([
    html.Div([dcc.Link(PropertyType[property_name].value.display_names[0],
                       href='/property/{}'.format(property_name)),
             html.Br()])
    for property_name in all_property_names
])

properties_index = html.Div([
    html.H5('Currently supported properties:'),
    list_of_properties,
    html.Br(),
    dcc.Link('< Back', href='/')
])