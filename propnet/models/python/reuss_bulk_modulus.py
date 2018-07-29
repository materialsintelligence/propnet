

def plug_in(self, symbol_values):
    s = symbol_values['S']
    return {'B': 1/(s[0][0] + s[1][1] + s[2][2] + 2*(s[0][1] + s[1][2] + s[0][2]))}

config = {
    "name": "reuss_bulk_modulus",
    "connections": [
        {
            "inputs": [
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
        "S": "compliance_tensor_voigt",
        "B": "bulk_modulus"
    },
    "description": "\nModel calculating a lower bound for Bulk Modulus based on the Reuss calculation, derived from the full elastic\ntensor.",
    "references": [],
    "plug_in": plug_in
}