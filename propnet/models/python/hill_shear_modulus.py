import numpy as np


def plug_in(symbol_values):
    s = symbol_values['S']
    c = symbol_values['C']
    g1 = 15. / (8. * s[:3, :3].trace() -
                 4. * np.triu(s[:3, :3]).sum() +
                 3. * s[3:, 3:].trace())
    g2 = 1 / 15 * (c[0][0] + c[1][1] + c[2][2]) - 1 / 15 * (c[0][1] + c[1][2] + c[2][0]) + \
                 1 / 5 * (c[3][3] + c[4][4] + c[5][5])
    return {'G': 1/2*(g1+g2)}

config = {
    "name": "hill_shear_modulus",
    "connections": [
        {
            "inputs": [
                "C",
                "S"
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
        "S": "compliance_tensor_voigt",
        "G": "shear_modulus"
    },
    "description": "\nModel calculating an average value for Shear Modulus based on the\nReuss and Voigt calculation, derived from the full elastic tensor.",
    "references": [],
    "plug_in": plug_in
}
