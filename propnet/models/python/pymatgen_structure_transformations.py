from pymatgen.transformations.standard_transformations import AutoOxiStateDecorationTransformation


def plug_in(symbol_values):
    s = symbol_values['s']
    trans = AutoOxiStateDecorationTransformation()
    s_oxi = trans.apply_transformation(s)
    return {
        's_oxi': s_oxi
    }


DESCRIPTION = """
This model attempts to work out what oxidation state is on each 
crystallographic site using the materials analysis code pymatgen.
"""

config = {
    "name": "pymatgen_structure_transformations",
    "connections": [
        {
            "inputs": [
                "s"
            ],
            "outputs": [
                "s_oxi"
            ]
        }
    ],
    "categories": [
        "pymatgen",
        "transformations"
    ],
    "variable_symbol_map": {
        "s": "structure",
        "s_oxi": "structure_oxi"
    },
    "description": DESCRIPTION,
    "references": ["doi:10.1016/j.commatsci.2012.10.028"],
    "implemented_by": [
        "mkhorton"
    ],
    "plug_in": plug_in
}
