from gbml.elasticity import predict_k_g_from_entry


def plug_in(symbol_values):
    computed_entry = symbol_values['computed_entry']
    entry = {
        'material_id': '',
        'energy_per_atom': computed_entry.energy_per_atom,
        'is_hubbard': computed_entry.parameters['is_hubbard'],
        'nsites': symbol_values['nsites'],
        'pretty_formula': symbol_values['formula'],
        'volume': symbol_values['volume']
    }
    K, G, caveats = predict_k_g_from_entry(entry)
    return {
        'K': K,
        'G': G
    }


DESCRIPTION = """
This model uses Gradient Boosting Machine-Locfit (GBML) to give
predictions for bulk and shear moduli given material descriptors 
and training data.
"""

config = {
    "name": "gbml",
    "connections": [
        {
            "inputs": [
                "computed_entry",
                "volume",
                "nsites",
                "formula"
            ],
            "outputs": [
                "K",
                "G"
            ]
        }
    ],
    "categories": [
        "mechanical",
        "machine learning"
    ],
    "variable_symbol_map": {
        "K": "bulk_modulus",
        "G": "shear_modulus",
        "computed_entry": "computed_entry",
        "volume": "volume_unit_cell",
        "nsites": "nsites",
        "formula": "formula"
    },
    "description": DESCRIPTION,
    "references": [],
    "implemented_by": [
        "mkhorton"
    ],
    "plug_in": plug_in
}
