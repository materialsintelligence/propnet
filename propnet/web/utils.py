import logging

from os import path
from io import StringIO

from pybtex.database.input.bibtex import Parser
from pybtex.plugin import find_plugin

from propnet.symbols import DEFAULT_SYMBOL_TYPE_NAMES, Symbol
from propnet.models import DEFAULT_MODEL_NAMES
from propnet.core.models import AbstractModel

from monty.serialization import loadfn

log = logging.getLogger(__name__)
AESTHETICS = loadfn(path.join(path.dirname(__file__), 'aesthetics.yaml'))


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

        id = None

        # should do better parsing of nodes here
        # TODO: this is also horrific code for demo, change
        if isinstance(n, Symbol):
            # property
            id = n.name
            label = n.display_names[0]
            node_type = 'Symbol'
        elif isinstance(n, AbstractModel):
            # model
            id = n.__class__.__name__
            label = n.title
            node_type = 'Model'
        if id:
            # Get node, labels, id, and title
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

            node['id'] = id
            nodes.append(node)

    log.info("Nodes to highlight green: {}".format(
            nodes_to_highlight_green))
    if nodes_to_highlight_green or nodes_to_highlight_yellow or nodes_to_highlight_red:
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
    for n1, n2 in graph.edges():

        id_n1 = None
        if isinstance(n1, Symbol):
            id_n1 = n1.name
        elif isinstance(n1, AbstractModel):
            id_n1 = n1.__class__.__name__

        id_n2 = None
        if isinstance(n2, Symbol):
            id_n2 = n2.name
        elif isinstance(n2, AbstractModel):
            id_n2 = n2.__class__.__name__

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
    elif pathname.startswith('/graph'):
        mode = 'graph'

    return {
        'mode': mode,
        'value': value
    }
