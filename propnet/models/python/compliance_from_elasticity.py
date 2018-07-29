import numpy as np


def plug_in(self, symbol_values):
    if 'C' in symbol_values.keys():
        c = symbol_values['C']
        return {'S': np.linalg.inv(c)}
    elif 'S' in symbol_values.keys():
        s = symbol_values['S']
        return {'C': np.linalg.inv(s)}

config = {
    "name": "compliance_from_elasticity",
    "connections": [
        {
            "inputs": [
                "C"
            ],
            "outputs": [
                "S"
            ]
        },
        {
            "inputs": [
                "S"
            ],
            "outputs": [
                "C"
            ]
        }
    ],
    "categories": [
        "mechanical"
    ],
    "symbol_property_map": {
        "C": "elastic_tensor_voigt",
        "S": "compliance_tensor_voigt"
    },
    "description": "\nModel calculating the compliance / elastic tensors from the elastic / compliance tensor.\nThis is a simple matrix inverse operation in voigt notation.",
    "references": [],
    "plug_in": plug_in
}