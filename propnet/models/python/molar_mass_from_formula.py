from pymatgen import Composition

def plug_in(symbol_values):
    f = symbol_values['formula']
    return {"molar_mass": Composition(f).weight}


DESCRIPTION = """
Simple conversion from formula to molar mass
"""

config = {
    "name": "clarke_thermal_conductivity",
    "connections": [
        {
            "inputs": [
                "formula",
            ],
            "outputs": [
                "molar_mass"
            ]
        }
    ],
    "categories": [
    ],
    "description": DESCRIPTION,
    "references": [],
    "plug_in": plug_in
}
