import dash_core_components as dcc
import dash_html_components as html


def home_layout():
    layout = html.Div(
        [html.Div("Not intended for public use at this time",
                  style={'color': 'red', 'font-weight': 'bold', 'text-align': 'center'}),
         html.Div("Internal collaborators only. Please report bugs :)",
                  style={'text-align': 'center'}),
         html.Br(),
         html.Details([
             html.Summary(html.H5("What is propnet?",
                                  style={'display': 'inline-block',
                                         'vertical-align': 'middle'})),
             dcc.Markdown("""
Property data and relationships between properties are the fundamental bases of scientific knowledge. 
These building blocks connect to form a web or graph of knowledge that can traverse different scientific domains.
_propnet_ is a Python-based software package which allows users to explore and expand datasets of materials properties 
using fundamental and literature-sourced relationships between properties across knowledge domains.

In _propnet_, we have codified these properties and relationships with the following goals:
"""),
             html.Ul([
                 html.Li("Connect knowledge domains through common properties",
                         style={'margin-bottom': 0}),
                 html.Li(
                     "Uncover \"hidden\" information in datasets by deriving " +
                     "new properties from known relationships and data",
                     style={'margin-bottom': 0}),
                 html.Li("Explore new relationships and correlations between properties",
                         style={'margin-bottom': 0})])
         ]),
         html.Details([
             html.Summary(html.H5("What is on this website?",
                                  style={'display': 'inline-block',
                                         'vertical-align': 'middle'})),
             dcc.Markdown("""
The purpose of this site is to demonstrate some of the features available in the _propnet_ package:
"""),
             html.Ul([
                 html.Li(
                     html.P([
                         html.B("Explore"), " allows you to see what properties (symbols) and "
                                            "models are currently built into ", html.Em("propnet"),
                         " (as of the last website build)."],
                         style={'display': 'inline'}),
                     style={'margin-bottom': 0}),
                 html.Li(
                     html.P([
                         html.B("Generate"), " allows you to run the graph traversal algorithm "
                                             "on your own data or data imported from ",
                         html.A("Materials Project", href="https://materialsproject.org"), " or ",
                         html.A("AFLOW", href="http://aflowlib.org"), "."],
                         style={'display': 'inline'}),
                     style={'margin-bottom': 0}),
                 html.Li(
                     html.P([
                         html.B("Correlate"), " allows you to explore different metrics used to "
                                              "correlate scalar properties calculated from MP data."],
                         style={'display': 'inline'}),
                     style={'margin-bottom': 0}),
                 html.Li(
                     html.P([
                         html.B("Plot"), " allows you to plot two, three, or four scalar "
                                         "properties against one another. This information pairs with Correlate data."],
                         style={'display': 'inline'}),
                     style={'margin-bottom': 0}),
                 html.Li(
                     html.P([
                         html.B("References"), " shows citations for our database data and for some of "
                                               "our key analysis/retrieval packages."],
                         style={'display': 'inline'}),
                     style={'margin-bottom': 0})
             ])
         ]),
         html.Details([
             html.Summary(html.H5("How can I apply propnet to my own data?",
                                  style={'display': 'inline-block',
                                         'vertical-align': 'middle'})),
             dcc.Markdown("""
A design goal of this project is to make it easily accessible for scientists who aren't coding experts.
Please see our [demo notebook](https://github.com/materialsintelligence/propnet/blob/master/demo/Getting%20Started.ipynb)
for a step-by-step example of how to apply _propnet_ to your own data. If you have a Materials Project API key,
you can also apply _propnet_ to an MP material.
""")]),
         html.Details([
             html.Summary(html.H5("How powerful is propnet?",
                                  style={'display': 'inline-block',
                                         'vertical-align': 'middle'})),
             dcc.Markdown("""
To demonstrate the power of propnet, we connected it to the materials in the Materials Project database.
Applying our graph traversal algorithm, we were able to expand the total number of known properties 
by over 300%.

As the code base grows with more models and properties, more data can be derived! We welcome suggestions or
contributions of models and properties (symbols). Please contact us via our GitHub page (linked below).
""")]),
         html.Details([
             html.Summary(html.H5("Where can I download the code or make a suggestion?",
                                  style={'display': 'inline-block',
                                         'vertical-align': 'middle'})),
             dcc.Markdown("""
Our GitHub repository is available [here](https://github.com/materialsintelligence/propnet).
If you would like to test _propnet_ locally, please clone the repository, `pip install -r requirements.txt`
and `python setup.py develop`.

If you have a suggestion for a feature or for a model, please open an issue or contact the team via GitHub.
""")]),
         html.Br(),
         html.Div(
             "Funded by Accelerated Materials Design and Discovery, Toyota Research Institute",
             style={'text-align': 'center',
                    'font-size': "1rem"}
         )]
    )
    return layout
