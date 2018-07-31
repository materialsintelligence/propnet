from pymatgen.analysis.find_dimension import find_dimension
from pymatgen.analysis.structure_analyzer import get_dimensionality


def plug_in(symbol_values):
    structure = symbol_values['structure']
    return {
        'dimensionality_cheon': find_dimension(structure),
        'dimensionality_gorai': get_dimensionality(structure)
    }

DESCRIPTION = """
Calculates the dimensionality of a structure using one of two methods 
implemented in pymatgen.
"""

config = {
    "name": "dimensionality",
    "connections": [
        {
            "inputs": [
                "structure"
            ],
            "outputs": [
                "dimensionality_cheon",
                "dimensionality_gorai"
            ]
        }
    ],
    "categories": [
        "structure"
    ],
    "symbol_property_map": {
        "dimensionality_cheon": "dimensionality",
        "dimensionality_gorai": "dimensionality",
        "structure": "structure"
    },
    "description": DESCRIPTION,
    "references": ["doi:10.1039/C6TA04121C", "doi:10.1021/acs.nanolett.6b05229"],
    "plug_in": plug_in
}
