import dash_html_components as html
import dash_core_components as dcc

from propnet.models import all_model_names

def model_layout(model_name):
    return html.Div([
        html.H3(model_name)
    ])

list_models = html.Div([
    html.Link(model_name, href='{}'.format(model_name))
    for model_name in all_model_names
])