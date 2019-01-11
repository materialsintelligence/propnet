def plug_in(symbol_values):
    c = symbol_values['C']
    return {'B': 1/9*(c[0][0] + c[1][1] + c[2][2]) +
                 2/9*(c[0][1] + c[1][2] + c[0][2])}


DESCRIPTION = """
Calculating relative permittivity and optical index from dielectric tensor
"""

config = {
    "name": "relative_permittivity",
    "connections": [
        {
            "inputs": [
                "C"
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
        "B": "bulk_modulus"
    },
    "description": DESCRIPTION,
    "references": [],
    "implemented_by": [
        "dmrdjenovich"
    ],
    "plug_in": plug_in
}
