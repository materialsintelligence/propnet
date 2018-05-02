import numpy as np

from os import path
from enum import Enum
from io import StringIO
from json import dumps

from pybtex.database.input.bibtex import Parser
from pybtex.plugin import find_plugin
from pybtex.style.labels import BaseLabelStyle

from propnet.symbols import DEFAULT_SYMBOL_TYPE_NAMES, Symbol
from propnet.models import DEFAULT_MODEL_NAMES

from monty.serialization import loadfn

AESTHETICS = loadfn(path.join(path.dirname(__file__), 'aesthetics.yaml'))


def graph_conversion(graph, highlight=False,
                     highlight_green=(),
                     highlight_yellow=(),
                     highlight_red=()):
    """Utility function to

    Args:
      graph: from Propnet.graph
      selected_node:  (Default value = None)
      highlighted_nodes:  (Default value = None)

    Returns:

    """

    nodes = []
    links = []

    # TODO: this utility function is a prototype, to be replaced!

    colors = {
        'property': 'orange',
        'model': 'blue',
        'condition': '#00ff00',
        'object': 'purple'
    }

    for n in graph.nodes():

        id = None

        # should do better parsing of nodes here
        # TODO: this is also horrific code for demo, change
        if n.node_type.name == 'Symbol':
            # property
            id = n.node_value.name
            label = n.node_value.display_names[0]
            fill = AESTHETICS['color'][n.node_value.category]
            shape = 'circle'
            radius = 5.0
        elif n.node_type.name == 'Model':
            # model
            id = n.node_value.__class__.__name__
            label = n.node_value.title
            fill = AESTHETICS['color']['model']
            shape = 'square'
            radius = 6.0

        if id:
            nodes.append({
                'id': id,
                'label': label,
                'fill': fill,
                'shape': shape,
                'radius': radius
            })

    if highlight:
        for node in nodes:
            if node['id'] in highlight_green:
                node['fill'] = '#9CDC90'
            elif node['id'] in highlight_yellow:
                node['fill'] = '#FFBF00'
            elif node['id'] in highlight_red:
                node['fill'] = '#FD9998'
            else:
                node['fill'] = '#BDBDBD'

    for n1, n2 in graph.edges():

        id_n1 = None
        if n1.node_type.name == 'Symbol':
            id_n1 = n1.node_value.name
        elif n1.node_type.name == 'Model':
            id_n1 = n1.node_value.__class__.__name__

        id_n2 = None
        if n2.node_type.name == 'Symbol':
            id_n2 = n2.node_value.name
        elif n2.node_type.name == 'Model':
            id_n2 = n2.node_value.__class__.__name__

        if id_n1 and id_n2:
            links.append({
                'source': id_n1,
                'target': id_n2,
                'value': 1.0
            })

    graph_data = {
        'nodes': nodes,
        'links': links
    }

    return dumps(graph_data)


def references_to_markdown(references):
    """Utility function to convert a BibTeX string containing
    references into a Markdown string.

    Args:
      references: BibTeX string

    Returns:
      Markdown string

    """

    pybtex_style = find_plugin('pybtex.style.formatting', 'plain')()
    pybtex_md_backend = find_plugin('pybtex.backends', 'markdown')
    pybtex_parser = Parser()

    # hack to not print labels (may remove this later)
    def write_entry(self, key, label, text):
        """

        Args:
          key: 
          label: 
          text: 

        Returns:

        """
        self.output(u'%s  \n' % text)
    pybtex_md_backend.write_entry = write_entry
    pybtex_md_backend = pybtex_md_backend()

    data = pybtex_parser.parse_stream(StringIO(references))
    data_formatted = pybtex_style.format_entries(data.entries.itervalues())
    output = StringIO()
    pybtex_md_backend.write_to_stream(data_formatted, output)

    # add blockquote style
    references_md = '> {}'.format(output.getvalue())
    references_md.replace('\n', '\n> ')

    return references_md

def uri_to_breadcrumb_layout(uri):
    """

    Args:
      uri: return:

    Returns:

    """
    return


def parse_path(pathname):
    """Utility function to parse URL path for routing purposes etc.
    This function exists because the path has to be parsed in
    a few places for callbacks.

    Args:
      path: path from URL
      pathname: 

    Returns:
      dictionary containing 'mode' ('property', 'model' etc.),
      'value' (name of property etc.)

    """

    if pathname == '/' or pathname is None:
        return None

    mode = None  # 'property' or 'model'
    value = None  # property name / model name

    if pathname == '/model':
        mode = 'model'
    elif pathname.startswith('/model'):
        mode = 'model'
        for model in DEFAULT_MODEL_NAMES:
            if pathname.startswith('/model/{}'.format(model)):
                value = model
    elif pathname == '/property':
        mode = 'property'
    elif pathname.startswith('/property'):
        mode = 'property'
        for property in DEFAULT_SYMBOL_TYPE_NAMES:
            if pathname.startswith('/property/{}'.format(property)):
                value = property
    elif pathname == '/load_material':
        mode = 'load_material'
    elif pathname.startswith('/load_material'):
        mode = 'load_material'
        value = pathname.split('/')[-1]

    return {
        'mode': mode,
        'value': value
    }
