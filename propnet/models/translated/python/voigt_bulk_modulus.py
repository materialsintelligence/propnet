

def plug_in(self, symbol_values):
    c = symbol_values['C']
    return {'B': 1/9*(c[0][0] + c[1][1] + c[2][2]) + 2/9*(c[0][1] + c[1][2] + c[0][2])}

config = {
    "name": "voigt_bulk_modulus",
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
    "description": "\nModel calculating an upper bound for Bulk Modulus based on\nthe Voigt calculation, derived from the full elastic tensor.",
    "references": [],
    "plug_in": plug_in
}