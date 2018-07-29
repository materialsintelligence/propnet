

def plug_in(symbol_values):
    c = symbol_values['C']
    return {'G': 1 / 15 * (c[0][0] + c[1][1] + c[2][2]) - 1 / 15 * (c[0][1] + c[1][2] + c[2][0]) +
                 1 / 5 * (c[3][3] + c[4][4] + c[5][5])}

config = {
    "name": "voigt_shear_modulus",
    "connections": [
        {
            "inputs": [
                "C"
            ],
            "outputs": [
                "G"
            ]
        }
    ],
    "categories": [
        "mechanical"
    ],
    "symbol_property_map": {
        "C": "elastic_tensor_voigt",
        "G": "shear_modulus"
    },
    "description": "\nModel calculating an upper bound for Shear Modulus based on\nthe Voigt calculation, derived from the full elastic tensor.",
    "references": [],
    "plug_in": plug_in
}
