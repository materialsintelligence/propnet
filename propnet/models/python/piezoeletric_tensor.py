import numpy as np


def plug_in(symbol_values):
    data = {}
    if "e" in symbol_values and "S" in symbol_values:
        data["d"] = np.einsum("ij,jk->ki", symbol_values["e"], symbol_values["S"])
    elif "d" in symbol_values and "C" in symbol_values:
        data["e"] = np.einsum("ij,jk->ik", symbol_values["d"], symbol_values["C"])

    return data


config = {
    "name": "piezoelectric_tensor",
    "connections": [{
        "inputs": ["e", "S"],
        "outputs": ["d"]
    }, {
        "inputs": ["d", "C"],
        "outputs": ["e"]
    }],
    "categories": ["mechanical", "electrical"],
    "variable_symbol_map": {
        "C": "elastic_tensor_voigt",
        "S": "compliance_tensor_voigt",
        "e": "piezoelectric_tensor",
        "d": "piezoelectric_tensor_converse"
    },
    "description": "Model relating the direct and converse piezoelectric tensor via the elastic tensor",
    "references": [],
    "implemented_by": [
        "shyamd"
    ],
    "plug_in": plug_in
}
