import dash
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output
from propnet import log_stream
from propnet.web.app_graph import graph_component
from propnet.web.app_model import model_layout, list_models
from propnet.web.app_property import property_layout, list_properties
from propnet.models import all_model_names
from propnet.properties import all_property_names

app = dash.Dash()
server = app.server
app.config.supress_callback_exceptions = True

d3 = "https://d3js.org/d3.v4.min.js"
app.scripts.append_script({"external_url": d3})

app.layout = html.Div(children=[
    html.H1(children='Propnet'),
    html.H3(children='A knowledge graph for materials science'),
    html.Br(),
    graph_component,
    html.Br(),
    dcc.Link('All Models', href='/model'),
    html.Br(),
    dcc.Link('All Properties', href='/property'),
    html.Br(),
    dcc.Location(id='url', refresh=False),
    dcc.Markdown('''```
Log:
{}
```'''.format(log_stream.getvalue()))
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/model':
        return list_models
    elif pathname.startswith('/model'):
        for model in all_model_names:
            if pathname == ('/model/{}'.format(model)):
                return model_layout(model)
    elif pathname == '/property':
        return list_properties
    elif pathname.startswith('/property'):
        for property in all_property_names:
            if pathname == ('/property/{}'.format(property)):
                return property_layout(property)
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=False)