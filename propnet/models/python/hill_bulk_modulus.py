def plug_in(symbol_values):
    s = symbol_values['S']
    c = symbol_values['C']
    b1 = 1 / (s[0][0] + s[1][1] + s[2][2] + 2 * (s[0][1] + s[1][2] + s[0][2]))
    b2 = 1 / 9 * (c[0][0] + c[1][1] + c[2][2]) + 2 / 9 * (c[0][1] + c[1][2] + c[0][2])
    return {'B': 1/2 * (b1+b2)}


DESCRIPTION = """
Model calculating an average value for Bulk Modulus based on the
Reuss and Voigt calculation, derived from the full elastic tensor.
"""

config = {
    "name": "hill_bulk_modulus",
    "connections": [
        {
            "inputs": [
                "C",
                "S"
            ],
            "outputs": [
                "B"
            ]
        }
    ],
    "categories": [
        "mechanical"
    ],
    "symbol_property_map": {
        "C": "elastic_tensor_voigt",
        "S": "compliance_tensor_voigt",
        "B": "bulk_modulus"
    },
    "description": DESCRIPTION,
    "references": [],
    "implemented_by": [
        "dmrdjenovich"
    ],
    "plug_in": plug_in
}
