import dash_html_components as html
import dash_core_components as dcc

from propnet.properties import all_property_names

def property_layout(property_name):
    return html.Div([
        html.H3(property_name)
    ])


list_properties = html.Div([
    html.Link(property_name, href='{}'.format(property_name))
    for property_name in all_property_names
])