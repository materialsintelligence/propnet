import numpy as np


def plug_in(symbol_values):
    s = symbol_values['S']
    return {'G': 15. / (8. * s[:3, :3].trace() -
                 4. * np.triu(s[:3, :3]).sum() +
                 3. * s[3:, 3:].trace())}


DESCRIPTION = """
Model calculating a lower bound for Shear Modulus based on the Reuss 
calculation, derived from the full elastic tensor
"""

config = {
    "name": "reuss_shear_modulus",
    "connections": [
        {
            "inputs": [
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
        "S": "compliance_tensor_voigt",
        "G": "shear_modulus"
    },
    "description": DESCRIPTION,
    "references": [],
    "plug_in": plug_in
}
