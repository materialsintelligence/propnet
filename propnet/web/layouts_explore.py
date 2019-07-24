import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate

from dash.dependencies import Input, Output, State

from dash_cytoscape import Cytoscape
from propnet.web.utils import graph_conversion, GRAPH_STYLESHEET, \
    GRAPH_LAYOUT_CONFIG, GRAPH_SETTINGS, propnet_nx_graph, update_labels


from propnet.web.layouts_models import models_index
from propnet.web.layouts_symbols import symbols_index
from propnet.core.registry import Registry

models_to_show = [
        'cost',
        'hhi',
        'magnetization_normalized_volume',
        'molar_mass_from_formula',
        'gbml',
        'density_relations',
        'pymatgen_structure_properties',
        'clarke_thermal_conductivity',
        'voigt_bulk_modulus',
        'hill_bulk_modulus',
        'reuss_bulk_modulus',
        'compliance_from_elasticity',
        'piezoelectric_tensor',
        'electromechanical_coupling',
        'homogeneous_elasticity_relations',
        'debye_temperature',
        'sound_velocity_elastic_longitudinal',
        'sound_velocity_elastic_transverse',
        'sound_velocity_elastic_mean'
    ]
symbols_to_show = [
        'hhi_production',
        'hhi_reserve',
        'cost_per_kg',
        'cost_per_mol',
        'total_magnetization_per_volume',
        'total_magnetization',
        'molar_mass',
        'volume_unit_cell',
        'volume_per_atom',
        'lattice',
        'composition',
        'nsites',
        'mass_per_atom',
        'density',
        'computed_entry',
        'formula',
        'youngs_modulus',
        'bulk_modulus',
        'compliance_tensor_voigt',
        'elastic_tensor_voigt',
        'piezoelectric_tensor',
        'piezoelectric_tensor_converse',
        'electromechanical_coupling',
        'thermal_conductivity',
        'debye_temperature',
        'sound_velocity_longitudinal',
        'sound_velocity_transverse',
        'sound_velocity_mean',
    ]
labels = [Registry("models")[v] for v in models_to_show] + [Registry("symbols")[v] for v in symbols_to_show]

def explore_layout(app):
    # graph_data = graph_conversion(propnet_nx_graph, hide_unconnected_nodes=False)

    graph_data = graph_conversion(propnet_nx_graph, hide_unconnected_nodes=True,
                                  labels_to_show=labels,
                                  show_symbol_labels=False,
                                  show_model_labels=False)
    graph_component = html.Div(
        id='graph_component',
        children=[Cytoscape(id='pn-graph', elements=graph_data,
                            stylesheet=GRAPH_STYLESHEET,
                            layout=GRAPH_LAYOUT_CONFIG,
                            **GRAPH_SETTINGS['full_view'])],
        )

    graph_layout = html.Div(
        id='graph_top_level',
        children=[
            html.Div([
                dcc.Checklist(id='graph_options',
                              options=[{'label': 'Show models',
                                        'value': 'show_models'},
                                       {'label': 'Show properties',
                                        'value': 'show_properties'}],
                              value=['show_properties'],
                              labelStyle={'display': 'inline-block'},
                              style={'display': 'inline-block'}),
                html.Button('Download PNG', id='download-png',
                            style={'display': 'inline-block',
                                   'margin-left': '10px'})
            ]),
            html.Div(id='graph_explorer',
                     children=[graph_component])])

    @app.callback(Output('pn-graph', 'elements'),
                  [Input('graph_options', 'value')],
                  [State('pn-graph', 'elements')])
    def change_propnet_graph_label_selection(props, elements):
        show_properties = 'show_properties' in props
        show_models = 'show_models' in props

        # update_labels(elements, show_models=show_models, show_symbols=show_properties)
        update_labels(elements, show_models=show_models, show_symbols=show_properties,
                      models_to_show=models_to_show, symbols_to_show=symbols_to_show)

        return elements

    @app.callback(Output('pn-graph', 'generateImage'),
                  [Input('download-png', 'n_clicks')])
    def download_image(n_clicks):
        if n_clicks is None:
            raise PreventUpdate
        return {
            'type': 'png',
            'action': 'download',
            'filename': 'pngraph'
        }

    layout = html.Div([
        html.Div([graph_layout], className='row'),
        html.Div([
            html.Div([models_index], className='six columns'),
            html.Div([symbols_index()], className='six columns')],
            className='row'),
        html.Div(id='emptydiv')
    ])

    return layout
