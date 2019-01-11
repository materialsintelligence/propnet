import numpy as np


def plug_in(symbol_values):

    req_symbols = ["S", "e", "d"]
    if all(s in symbol_values for s in req_symbols):
        e = data["e"]
        S = data["S"]
        d = data["d"]

        data["k"] = d[3][3] / np.sqrt(e[3][3] * S[3][3])

    return data


DESCRIPTION = """
Model calculating the electromechanical coupling factor,
which is the efficiency of converting eletrical energy
to acoustic energy in a piezoeletric transducer or filter
"""

config = {
    "name": "eletromechanical_coupling",
    "connections": [{
        "inputs": ["e", "S"],
        "outputs": ["d"]
    }, {
        "inputs": ["d", "C"],
        "outputs": ["e"]
    }],
    "categories": ["mechanical", "electrical"],
    "symbol_property_map": {
        "S": "compliance_tensor_voigt",
        "e": "dieletric_tensor",
        "d": "piezoelectric_tensor_converse",
        "k": "electromechanical_coupling"
    },
    "description": DESCRIPTION,
    "implemented_by": ["shyamd"],
    "references": [],
    "plug_in": plug_in
}
