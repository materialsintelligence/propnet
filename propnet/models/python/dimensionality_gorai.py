from pymatgen.analysis.dimensionality import get_dimensionality_gorai


def plug_in(symbol_values):
    structure = symbol_values['structure']
    return {
        'dimensionality': get_dimensionality_gorai(structure)
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
    "implemented_by": [
        "mkhorton"
    ],
    "plug_in": plug_in
}
