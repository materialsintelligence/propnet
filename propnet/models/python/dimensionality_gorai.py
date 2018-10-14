from pymatgen.analysis.structure_analyzer import get_dimensionality


def plug_in(symbol_values):
    structure = symbol_values['structure']
    return {
        'dimensionality': get_dimensionality(structure)
    }

DESCRIPTION = """
Calculates the dimensionality of a structure using one of two methods 
implemented in pymatgen.
"""

config = {
    "name": "dimensionality_gorai",
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
    "references": ["doi:10.1021/acs.nanolett.6b05229"],
    "plug_in": plug_in
}
