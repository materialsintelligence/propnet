import dash
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output

from propnet.web.layouts_models import model_layout, models_index
from propnet.web.layouts_symbols import symbol_layout, symbols_index
from propnet.web.layouts_plot import get_plot_layout, define_plot_callbacks
from propnet.web.layouts_home import home_layout
from propnet.web.layouts_interactive import interactive_layout
from propnet.web.layouts_correlate import correlate_layout
from propnet.web.layouts_explore import explore_layout
from propnet.web.layout_refs import refs_layout

from dash_cytoscape import load_extra_layouts

from propnet.web.utils import parse_path

from flask_caching import Cache
import logging

log = logging.getLogger(__name__)

load_extra_layouts()

# TODO: Fix math rendering

app = dash.Dash(__name__)
server = app.server
app.config.suppress_callback_exceptions = True  # TODO: remove this?
app.scripts.config.serve_locally = True
app.title = "propnet"
route = dcc.Location(id='url', refresh=False)

cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem', 'CACHE_DIR': '.tmp'
})


layout_menu = html.Div(
    children=[dcc.Link('What is propnet?', href='/home'),
              html.Span(' • '),
              dcc.Link('Explore', href='/explore'),
              html.Span(' • '),
              dcc.Link('Generate', href='/generate'),
              html.Span(' • '),
              dcc.Link('Correlate', href='/correlate'),
              html.Span(' • '),
              dcc.Link('Plot', href='/plot'),
              html.Span(' • '),
              dcc.Link('References', href='/refs')
              ])
# header
app.layout = html.Div(
    children=[route,
              html.Div([html.H3(app.title), layout_menu, html.Br()],
                       style={'textAlign': 'center'}),
              html.Div(id='page-content')],
    style={'marginLeft': 200, 'marginRight': 200, 'marginTop': 30})

CORRELATE_LAYOUT = correlate_layout(app)
INTERACTIVE_LAYOUT = interactive_layout(app)
EXPLORE_LAYOUT = explore_layout(app)
REFS_LAYOUT = refs_layout(app)
define_plot_callbacks(app)

# routing, current routes defined are:
# / for home page
# /model for model summary
# /model/model_name for information on that model
# /property for property summary
# /property/property_name for information on that property


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname'),
               Input('url', 'search')])
def display_page(pathname, search):
    """

    Args:
      pathname:
      search:

    Returns:

    """

    path_info = parse_path(pathname, search)

    if path_info:
        if path_info['mode'] == 'model':
            if path_info['value']:
                return model_layout(path_info['value'])
            else:
                return models_index
        elif path_info['mode'] == 'property':
            if path_info['value']:
                property_name = path_info['value']
                return symbol_layout(property_name)
            else:
                return symbols_index()
        elif path_info['mode'] == 'explore':
            return EXPLORE_LAYOUT
        elif path_info['mode'] == 'plot':
            props = None
            if path_info['value'] is not None:
                props = path_info['value']
            return get_plot_layout(props)
        elif path_info['mode'] == 'generate':
            return INTERACTIVE_LAYOUT
        elif path_info['mode'] == 'correlate':
            return CORRELATE_LAYOUT
        elif path_info['mode'] == 'refs':
            return REFS_LAYOUT
        elif path_info['mode'] == 'home':
            return home_layout()
        else:
            return '404'
    else:
        return home_layout()


if __name__ == '__main__':
    app.run_server(debug=False, dev_tools_ui=True)
