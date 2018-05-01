from collections import OrderedDict

import dash_html_components as html
import dash_core_components as dcc

import propnet.models as models

from propnet.symbols import DEFAULT_SYMBOL_TYPES
from propnet.web.utils import references_to_markdown


# layouts for model detail pages

def model_image_component(model):
    """
    Get a Robohash of a provided model, to give a
    face to a name.

    Args:
      model: instance of an AbstractModel subclass

    Returns: a Dash Img element

    """
    url = "https://robohash.org/{}".format(model.__hash__())
    return html.Img(src=url, style={'width': 150, 'border-radius': '50%'})


def model_layout(model_name):
    """Create a Dash layout for a provided model.

    Args:
      model: an instance of an AbstractModel subclass

    Returns:
      Dash layout

    """

    # instantiate model from name
    model = getattr(models, model_name)()

    badge = html.Div(
        className='double-val-label',
        children=[
            html.Span('model id', className='model'),
            html.Span(str(model.uuid)[-6:])
        ]
    )

    model_title = html.Div(
        className='row',
        children=[
            html.Div(
                className='three columns',
                style={'text-align': 'right'},
                children=[
                    model_image_component(model)
                ]
            ),
            html.Div(
                className='nine columns',
                children=[
                    html.H3(model.title),
                    badge
                ]
            )
        ]
    )

    # dict to hold layouts for each section
    layouts = OrderedDict()

    if model.tags:
        tags = html.Ul(
            className="tags",
            children=[
                html.Li(tag, className="tag") for tag in model.tags
            ]
        )

        layouts['Tags'] = tags

    if model.references:
        references = html.Div([dcc.Markdown(references_to_markdown(ref))
                               for ref in model.references])

        layouts['References'] = references

    symbols = html.Div(
        children=[
            html.Div(
                className='row',
                children=[
                    html.Div(
                        className='two columns',
                        children=[
                            str(symbol)
                        ]
                    ),
                    html.Div(
                        className='ten columns',
                        children=[
                            dcc.Link(DEFAULT_SYMBOL_TYPES[symbol_name].display_names[0],
                                     href='/property/{}'.format(symbol_name))
                        ]
                    )
                ]
            )
            for symbol, symbol_name in model.symbol_mapping.items()
        ]
    )

    layouts['Symbols'] = symbols

    if hasattr(model, 'equations'):
        equations = html.Div(
            children=[
                html.Div(
                    className='row',
                    children=[equation])
                for equation in model.equations
            ]
        )

        layouts['Equations'] = equations

    layouts['Description'] = dcc.Markdown(model.description)

    if model.test_data:
        sample_data_header = html.Div(
            className='row',
            children=[
                html.Div(
                    className='five columns',
                    style={'text-align': 'center'},
                    children=[
                        html.H4('Input(s)')
                    ]
                ),
                html.Div(
                    className='two columns',
                    style={'text-align': 'center'},
                    children=[
                        html.H4('->')
                    ]
                ),
                html.Div(
                    className='five columns',
                    style={'text-align': 'center'},
                    children=[
                        html.H4('Output(s)')
                    ]
                )
            ]
        )

        #layouts['Sample Data'] = html.Div(children=[sample_data_header, *sample_data])

        layouts['Sample Code'] = dcc.Markdown('```\n{}```'.format(model._example_code))

    sublayouts = []
    for title, layout in layouts.items():
        sublayouts.append(html.H6(title))
        sublayouts.append(layout)

    return html.Div([
        model_title,
        html.Br(),
        *sublayouts
    ])


model_links = {}
for model_name in models.DEFAULT_MODEL_NAMES:
    # instantiate model from name
    model = getattr(models, model_name)()

    # group by tag
    tags = model.tags

    for tag in tags:
        if tag not in model_links:
            model_links[tag] = []

    for tag in tags:
        passes = model.test()
        passes = "✅" if passes else "❌"

        link_text = "{}".format(model.title)

        model_links[tag].append(
            html.Div([
                html.Span('{} '.format(passes)),
                dcc.Link(link_text,
                         href='/model/{}'.format(model_name)),
                html.Br()
            ])
        )

model_links_grouped = []
for tag, links in model_links.items():
    model_links_grouped += [
        html.H6(tag.title()),
        html.Div(links),
        html.Br()
    ]

models_index = html.Div([
    html.H5('Current models:'),
    html.Div(model_links_grouped),
    html.Br(),
    dcc.Link('< Back', href='/')
])
