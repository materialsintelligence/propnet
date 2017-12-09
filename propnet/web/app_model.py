import dash_html_components as html
import dash_core_components as dcc

from propnet.models import all_model_names, EmpiricalBandGapSetyawan

# layouts for model detail pages

def model_image_component(model):

    url = "https://robohash.org/{}".format(model.__hash__())
    return html.Img(src=url, style={'width': 150, 'border-radius':'50%'})

def model_layout(model_name):

    # for test
    model = EmpiricalBandGapSetyawan.EmpiricalBandGapSetyawan()

    title = html.Div(className='row', children=[
        html.Div(className='three columns', children=[model_image_component(model)]),
        html.Div(className='one columns', children=[]),
        html.Div(className='eight columns', children=[
            html.H3(model.title),
            html.H5('model-id: {}-rev{}'.format(model.__hash__(), model.revision),
                    style={'font-family': 'monospace'})])])

    if model.references:
        import io
        import six
        import pybtex.database.input.bibtex
        import pybtex.plugin

        pybtex_style = pybtex.plugin.find_plugin('pybtex.style.formatting', 'plain')()
        pybtex_html_backend = pybtex.plugin.find_plugin('pybtex.backends', 'markdown')()
        pybtex_parser = pybtex.database.input.bibtex.Parser()

        my_bibtex = "\n".join(model.references)

        data = pybtex_parser.parse_stream(six.StringIO(my_bibtex))
        data_formatted = pybtex_style.format_entries(six.itervalues(data.entries))
        output = io.StringIO()
        pybtex_html_backend.write_to_stream(data_formatted, output)
        md = output.getvalue()

        references = dcc.Markdown(md)
    else:
        # TODO: append to list instead ...
        references = None

    if model.tags:

        tags = html.Ul(className="tags", children=[
                html.Li(tag, className="tag") for tag in model.tags
        ])
    else:
        references = None

    return html.Div([
        title,
        html.Br(),
        tags,
        html.Br(),
        html.H6('References'),
        references
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