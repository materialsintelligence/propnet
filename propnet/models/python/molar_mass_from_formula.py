from pymatgen import Composition

def plug_in(symbol_values):
    f = symbol_values['formula']
    return {"molar_mass": Composition(f).weight}


DESCRIPTION = """
Simple conversion from formula to molar mass
"""

config = {
    "name": "molar_mass_from_formula",
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
    "implemented_by": [
        "montoyjh"
    ],
    "plug_in": plug_in
}
