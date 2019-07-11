def plug_in(symbol_values):
    return {'is_metallic': True if symbol_values['E_g'] <= 0 else False}


DESCRIPTION = """
This model returns true if band gap is zero.
"""

config = {
    "name": "is_metallic",
    "connections": [
        {
            "inputs": [
                "E_g"
            ],
            "outputs": [
                "is_metallic"
            ]
        }
    ],
    "categories": [
        "classifier"
    ],
    "variable_symbol_map": {
        "E_g": "band_gap",
        "is_metallic": "is_metallic"
    },
    "description": DESCRIPTION,
    "references": [],
    "implemented_by": [
        "mkhorton"
    ],
    "plug_in": plug_in
}
