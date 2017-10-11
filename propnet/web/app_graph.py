import dash_html_components as html
import dash_core_components as dcc

from propnet.core.graph import Propnet
from propnet.properties import PropertyType
from propnet.core.models import AbstractModel
from plotly.graph_objs import *

import networkx as nx

# generates graph visualization
# adapted from example given in plot.ly documentation

g = Propnet().graph
pos = nx.spring_layout(g)

dmin=1
ncenter=0
for n in pos:
    x, y = pos[n]
    d = (x - 0.5)**2+(y - 0.5)**2
    if d < dmin:
        ncenter = n
        dmin = d

edge_trace = Scatter(
    x=[],
    y=[],
    line=Line(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines'
)

for edge in g.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_trace['x'] += [x0, x1, None]
    edge_trace['y'] += [y0, y1, None]

node_trace = Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers',
    hoverinfo='text',
    marker=Marker(
        showscale=False,
        reversescale=True,
        color=[],
        size=10,
        colorbar={
            'thickness': 15,
            'title': 'Node connections',
            'xanchor': 'left',
            'titleside': 'right'
        },
        line={
            'width': 2
        }
    )
)

for node in g.nodes():
    x, y = pos[node]
    node_trace['x'].append(x)
    node_trace['y'].append(y)
    model_color = "#8F67CA"
    property_type_color = "#4F8EE6"
    if isinstance(node, PropertyType):
        node_trace['marker']['color'].append(property_type_color)
        node_trace['text'].append(node.value.display_names[0])
    elif issubclass(node, AbstractModel):
        node_trace['marker']['color'].append(model_color)
        node_trace['text'].append(node().title)

fig = Figure(
    data=Data([edge_trace, node_trace]),
    layout=Layout(
        title="",
        showlegend=False,
        hovermode='closest',
        xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)
    )
)

graph_component = dcc.Graph(figure=fig,
                            style={'height':300,
                                   'width':600,
                                   'display': 'block',
                                   'margin-left': 'auto',
                                   'margin-right': 'auto'},
                            id='propnet-graph',
                            config={'displayModeBar': False})