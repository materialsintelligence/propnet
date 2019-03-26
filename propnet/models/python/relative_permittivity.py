import numpy as np


def plug_in(symbol_values):
    diel = symbol_values['diel']
    return {'eps_r': np.average(np.diag(diel))}


config = {
    "name":
    "relative_permittivity",
    "connections": [{
        "inputs": ["diel"],
        "outputs": ["eps_r"]
    }],
    "categories": ["mechanical"],
    "variable_symbol_map": {
        "diel": "dielectric_tensor",
        "eps_r": "relative_permittivity"
    },
    "description":
    "Calculating relative permittivity from dielectric tensor",
    "references": [],
    "implemented_by": ["shyamd"],
    "plug_in":
    plug_in,
    "test_data": [{
        "inputs": {
            "diel": [[18.65, 0.00, 0.00], [-0.00, 18.65, 0.00], [-0.00, 0.00, 7.88]]
        },
        "outputs": {
            "eps_r": 15.06
        }
    }]
}
