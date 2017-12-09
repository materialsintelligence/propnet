import json

from enum import Enum

def graph_conversion(graph):
    """
    Utility function to
    :param graph: from Propnet.graph
    :return:
    """

    nodes = []
    links = []

    # TODO: this utility function is a prototype, to be replaced!

    for n in graph.nodes():

        # should do better parsing of nodes here
        if isinstance(n, Enum):
            # property
            id = n.name
            label = n.value.display_names[0]
            fill = 'orange'
            shape = 'circle'
        else:
            # model
            id = n.__name__
            label = n().title
            fill = 'blue'
            shape = 'square'

        nodes.append({
            'id': id,
            'label': label,
            'fill': fill,
            'shape': shape
        })

    for n1, n2 in graph.edges():

        if isinstance(n1, Enum):
            id_n1 = n1.name
        else:
            id_n1 = n1.__name__

        if isinstance(n2, Enum):
            id_n2 = n2.name
        else:
            id_n2 = n2.__name__

        links.append({
            'source': id_n1,
            'target': id_n2,
            'value': 10
        })

    graph_data = {
        'nodes': nodes,
        'links': links
    }

    return json.dumps(graph_data)