import dash_html_components as html
import dash_core_components as dcc

from propnet.models import all_model_names

# layouts for model detail pages

def model_layout(model_name):
    return html.Div([
        html.H3(model_name)
    ])

model_links = html.Div([
    html.Div([
        dcc.Link(model_name, href='/model/{}'.format(model_name)),
        html.Br()
    ])
    for model_name in all_model_names
])

list_models = html.Div([
    html.H5('Current models:'),
    model_links,
    html.Br(),
    dcc.Link('< Back', href='/')
])