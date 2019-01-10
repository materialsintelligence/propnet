from pymatgen.analysis.find_dimension import find_dimension


def plug_in(symbol_values):
    structure = symbol_values['structure']
    return {
        'dimensionality': find_dimension(structure),
    }

DESCRIPTION = """
Calculates the dimensionality of a structure using cheon's method 
implemented in pymatgen.
"""

config = {
    "name": "dimensionality_cheon",
    "connections": [
        {
            "inputs": [
                "structure"
            ],
            "outputs": [
                "dimensionality",
            ]
        }
    ],
    "categories": [
        "structure"
    ],
    "description": DESCRIPTION,
    "references": ["doi:10.1039/C6TA04121C"],
    "implemented_by": [
        "mkhorton"
    ],
    "plug_in": plug_in
}
