import logging

from os import path

from propnet.core.symbols import Symbol
from propnet.core.models import Model

from monty.serialization import loadfn
import networkx as nx

from propnet.core.graph import Graph
# noinspection PyUnresolvedReferences
import propnet.models
from propnet.core.registry import Registry

log = logging.getLogger(__name__)

GRAPH_CONFIG = loadfn(path.join(path.dirname(__file__), 'graph_config.yaml'))
GRAPH_STYLESHEET = loadfn(path.join(path.dirname(__file__), 'graph_stylesheet.yaml'))
GRAPH_HEIGHT_PX = 800
SUBGRAPH_HEIGHT_PX = 500
propnet_nx_graph = Graph().get_networkx_graph()


# TODO: use the attributes of the graph class, rather than networkx
def graph_conversion(graph: nx.DiGraph,
                     derivation_pathway=None,
                     hide_unconnected_nodes=True,
                     show_symbol_labels=True,
                     show_model_labels=False):
    """Utility function to render a networkx graph
    from Graph.graph for use in GraphComponent

    Args:
        graph (networkx.graph): from Graph.graph

    Returns: graph dict
    """

    nodes = []
    edges = {}

    for n in graph.nodes():

        # should do better parsing of nodes here
        # TODO: this is also horrific code for demo, change
        # TODO: more dumb crap related to graph
        if isinstance(n, Symbol):
            # property
            name = n.name
            label = n.display_names[0]
            node_type = 'symbol'
        elif isinstance(n, Model):
            # model
            name = n.title
            label = n.title
            node_type = 'model'
        else:
            name = None
            label = None
            node_type = None

        if name:
            # Get node, labels, name, and title
            node = {
                'data': {'id': name,
                         'label': label
                         },
                'locked': False,
                'classes': [node_type]
            }

            if (node_type == 'model' and show_model_labels) or \
                    (node_type == 'symbol' and show_symbol_labels):
                node['classes'].append('label-on')
            else:
                node['classes'].append('label-off')

            nodes.append(node)

    connected_nodes = set()

    # TODO: need to clean up after model refactor
    def get_node_id(node_):
        return node_.title if isinstance(node_, Model) else node_.name

    for n1, n2 in graph.edges():
        id_n1 = get_node_id(n1)
        id_n2 = get_node_id(n2)

        if id_n1 and id_n2:
            connected_nodes.add(id_n1)
            connected_nodes.add(id_n2)
            if (id_n2, id_n1) in edges:
                edges[(id_n2, id_n1)]['classes'].append('is-output')
            else:
                edges[(id_n1, id_n2)] = {
                    'data': {'source': id_n1, 'target': id_n2},
                    'classes': ['is-input']}

    if not hide_unconnected_nodes or not derivation_pathway:
        unconnected_edges = {
            (node['data']['id'], 'unattached_symbols'):
                {'data': {'source': node['data']['id'],
                          'target': 'unattached_symbols'},
                 'classes': ['is-output']}
            for node in nodes if node['data']['id'] not in connected_nodes}
        if unconnected_edges:
            edges.update(unconnected_edges)
            nodes.append({
                'data': {'id': 'unattached_symbols',
                         'label': "Unattached symbols"
                         },
                'locked': False,
                'classes': ['unattached', 'label-on']
            })
    else:
        nodes = [node for node in nodes if node['data']['id'] in connected_nodes]

    # For highlighting graph derivation
    if derivation_pathway:
        symbols_in = [get_node_id(s) for s in derivation_pathway['inputs']]
        symbols_out = [get_node_id(s) for s in derivation_pathway['outputs']]
        models_evaluated = [get_node_id(m)
                            for m in derivation_pathway['models']]

        symbol_nodes_in_path = set.union(set(symbols_in),
                                         set(symbols_out))

        for edge in edges.values():
            if (edge['data']['source'] in symbol_nodes_in_path and
                    edge['data']['target'] in models_evaluated) or \
                    (edge['data']['target'] in symbol_nodes_in_path and
                     edge['data']['source'] in models_evaluated):
                edge['classes'].append('on-derivation-path')

        for node in nodes:
            node_id = node['data']['id']
            if node_id in symbols_in:
                node['classes'].append('symbol-input')
            elif node_id in symbols_out:
                node['classes'].append('symbol-derived')
            elif node_id in models_evaluated:
                node['classes'].append('model-derived')
            else:
                node['classes'].append('untraversed')

    for node in nodes:
        node['group'] = 'nodes'
    for edge in edges.values():
        edge['group'] = 'edges'

    graph_data = nodes + list(edges.values())

    for v in graph_data:
        if isinstance(v.get('classes'), list):
            v['classes'] = " ".join(v['classes'])

    return graph_data


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
        for property_ in Registry("symbols").keys():
            if pathname.startswith('/property/{}'.format(property_)):
                value = property_
    elif pathname.startswith('/explore'):
        mode = 'explore'
    elif pathname.startswith('/plot'):
        mode = 'plot'
    elif pathname.startswith('/generate'):
        mode = 'generate'
    elif pathname.startswith('/correlate'):
        mode = 'correlate'
    elif pathname.startswith('/home'):
        mode = 'home'

    return {
        'mode': mode,
        'value': value
    }


def update_labels(elements, show_models=True, show_symbols=True):
    for elem in elements:
        group = elem['group']
        if group == 'edge':
            # applies to nodes only
            continue
        classes = elem.get('classes')
        if not classes:
            # if there is no classes specified, not sure what it is otherwise
            continue
        classes_list = classes.split(" ")
        is_model = any(c.startswith("model") for c in classes_list)
        is_symbol = any(c.startswith("symbol") for c in classes_list)

        if not is_model and not is_symbol:
            # is some other element on the graph, like the "unattached" model
            continue

        class_to_add = 'label-off'
        if (is_model and show_models) or (is_symbol and show_symbols):
            class_to_add = 'label-on'

        for val in ('label-on', 'label-off'):
            try:
                classes_list.remove(val)
            except ValueError:
                pass
        classes_list.append(class_to_add)

        elem['classes'] = " ".join(classes_list)
