import logging

from os import path
from io import StringIO

from pybtex.database.input.bibtex import Parser
from pybtex.plugin import find_plugin

from propnet.core.symbols import Symbol
from propnet.core.models import Model

from monty.serialization import loadfn

import propnet.models
import propnet.symbols
from propnet.core.registry import Registry

log = logging.getLogger(__name__)
AESTHETICS = loadfn(path.join(path.dirname(__file__), 'aesthetics.yaml'))


# TODO: use the attributes of the graph class, rather than networkx
def graph_conversion(graph,
                     nodes_to_highlight_green=(),
                     nodes_to_highlight_yellow=(),
                     nodes_to_highlight_red=(),
                     hide_unconnected_nodes=True,
                     aesthetics=None):
    """Utility function to render a networkx graph
    from Graph.graph for use in GraphComponent

    Args:
      graph: from Graph.graph

    Returns: graph dict
    """

    aesthetics = aesthetics or AESTHETICS

    nodes = []
    edges = []
    log.info(aesthetics['node_aesthetics']['Symbol'])
    log.info(aesthetics['node_aesthetics']['Model'])
    for n in graph.nodes():

        name = None

        # should do better parsing of nodes here
        # TODO: this is also horrific code for demo, change
        # TODO: more dumb crap related to graph
        if isinstance(n, Symbol):
            # property
            name = n.name
            label = n.display_names[0]
            node_type = 'Symbol'
        elif isinstance(n, Model):
            # model
            name = n.title
            label = n.title
            node_type = 'Model'
        if name:
            # Get node, labels, name, and title
            node = aesthetics['node_aesthetics'][node_type].copy()
            if node.get("show_labels"):
                node.update({"label": label,
                             # "title": label,
                             "shape": "box",
                             "font": {"color": "#ffffff",
                                      "face": "Helvetica",
                                      "size": 20},
                             "size": 30})
            # Pop labels if they exist
            else:
                node.update({"size": 8.0,
                             "shape": "diamond",
                             "label": "",
                             "title": ""})

            node['id'] = name
            nodes.append(node)

    log.info("Nodes to highlight green: {}".format(
            nodes_to_highlight_green))
    highlight_nodes = any([nodes_to_highlight_green, nodes_to_highlight_yellow,
                           nodes_to_highlight_red])
    if highlight_nodes:
        log.debug("Nodes to highlight green: {}".format(
            nodes_to_highlight_green))
        for node in nodes:
            if node['id'] in nodes_to_highlight_green:
                node['color'] = '#9CDC90'
            elif node['id'] in nodes_to_highlight_yellow:
                node['color'] = '#FFBF00'
            elif node['id'] in nodes_to_highlight_red:
                node['color'] = '#FD9998'
            else:
                node['color'] = '#BDBDBD'

    connected_nodes = set()
    # TODO: need to clean up after model refactor
    def get_node_id(node):
        return node.title if isinstance(node, Model) else node.name

    for n1, n2 in graph.edges():
        id_n1 = get_node_id(n1)
        id_n2 = get_node_id(n2)

        if id_n1 and id_n2:
            connected_nodes.add(id_n1)
            connected_nodes.add(id_n2)
            edges.append({
                'from': id_n1,
                'to': id_n2
            })

    if hide_unconnected_nodes:
        for node in nodes:
            if node['id'] not in connected_nodes:
                node['hidden'] = True

    graph_data = {
        'nodes': nodes,
        'edges': edges
    }

    return graph_data


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


def parse_path(pathname):
    """Utility function to parse URL path for routing purposes etc.
    This function exists because the path has to be parsed in
    a few places for callbacks.

    Args:
      pathname (str): path from url

    Returns:
        (dict) dictionary containing 'mode' ('property', 'model' etc.),
        'value' (name of property etc.)

    """

    if pathname == '/' or pathname is None:
        return None

    mode = None  # 'property' or 'model'
    value = None  # property name / model name

    # TODO: get rid of this

    if pathname == '/model':
        mode = 'model'
    elif pathname.startswith('/model'):
        mode = 'model'
        for model in Registry("models").keys():
            if pathname.startswith('/model/{}'.format(model)):
                value = model
    elif pathname == '/property':
        mode = 'property'
    elif pathname.startswith('/property'):
        mode = 'property'
        for property in Registry("symbols").keys():
            if pathname.startswith('/property/{}'.format(property)):
                value = property
    elif pathname.startswith('/explore'):
        mode = 'explore'
    elif pathname.startswith('/plot'):
        mode = 'plot'
    elif pathname.startswith('/generate'):
        mode = 'generate'
    elif pathname.startswith('/home'):
        mode = 'home'

    return {
        'mode': mode,
        'value': value
    }
