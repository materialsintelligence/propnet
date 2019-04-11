import numpy as np


def plug_in(symbol_values):

    req_symbols = ["S", "e", "d"]
    data = {}
    if all(s in symbol_values for s in req_symbols):
        e = symbol_values["e"]
        S = symbol_values["S"]
        d = symbol_values["d"]

        data["k"] = np.abs(d[2][2] / np.sqrt(e[2][2] * S[2][2]))

    return data


DESCRIPTION = """
Model calculating the electromechanical coupling factor,
which is the efficiency of converting eletrical energy
to acoustic energy in a piezoeletric transducer or filter
"""

test_data = [{
    "inputs": {
        "S": [[0.007482236755310126, -0.002827041595205337, -0.002827041595205337, 0.0, 0.0, 0.0],
              [-0.002827041595205337, 0.007482236755310125, -0.002827041595205337, 0.0, 0.0, 0.0],
              [-0.0028270415952053366, -0.002827041595205337, 0.007482236755310125, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.010309278350515464, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.010309278350515464, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.010309278350515464]],
        "e": [[18.65, 0.00, 0.00], [-0.00, 18.65, 0.00], [-0.00, 0.00, 7.88]],
        "d": [[-0.0412497, -0.28686697, 0.06342802], [0.05065159, 0.26064878, -0.04828778],
              [0.08828203, 0.5660897, -0.11520665], [-0.16218673, -0.92468949, 0.2109461],
              [0.02485558, 0.03232004, -0.02421919], [0.06636329, 0.46541895, -0.09526407]]
    },
    "outputs": {
        "k": 0.47445902984
    }
}]

config = {
    "name": "eletromechanical_coupling",
    "connections": [{
        "inputs": ["e", "S", "d"],
        "outputs": ["k"]
    }],
    "categories": ["mechanical", "electrical"],
    "variable_symbol_map": {
        "S": "compliance_tensor_voigt",
        "e": "dielectric_tensor",
        "d": "piezoelectric_tensor_converse",
        "k": "electromechanical_coupling"
    },
    "description": DESCRIPTION,
    "implemented_by": ["shyamd"],
    "references": [],
    "plug_in": plug_in,
    "test_data": test_data
}
