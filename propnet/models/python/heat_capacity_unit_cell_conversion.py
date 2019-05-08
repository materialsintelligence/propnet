def plug_in(symbol_values):
    structure = symbol_values['structure']
    factor = structure.composition.get_reduced_formula_and_factor()[1]
    uc_cv = symbol_values.get("uc_cv")
    uc_cp = symbol_values.get("uc_cp")
    molar_cv = symbol_values.get("molar_cv")
    molar_cp = symbol_values.get("molar_cp")
    if uc_cv is not None:
        return {"molar_cv": uc_cv / factor * 6.022E23}
    if uc_cp is not None:
        return {"molar_cp": uc_cp / factor * 6.022E23}
    if molar_cv is not None:
        return {"uc_cv": molar_cv * factor / 6.022E23}
    if molar_cp is not None:
        return {"uc_cp": molar_cp * factor / 6.022E23}

DESCRIPTION = """
Properties of a crystal structure, such as the number of sites in its 
unit cell and its space group, as calculated by pymatgen.
"""

config = {
    "name": "heat_capacity_unit_cell_conversion",
    "connections": [
        {
            "inputs": [
                "structure",
                "uc_cv"
            ],
            "outputs": [
                "molar_cv",
            ]
        },
        {
            "inputs": [
                "structure",
                "uc_cp"
            ],
            "outputs": [
                "molar_cp",
            ]
        },
        {
            "inputs": [
                "structure",
                "molar_cv"
            ],
            "outputs": [
                "uc_cv",
            ]
        },
        {
            "inputs": [
                "structure",
                "molar_cp"
            ],
            "outputs": [
                "uc_cp",
            ]
        }
    ],
    "categories": [],
    "variable_symbol_map": {
        "structure": "structure",
        "uc_cv": "unit_cell_heat_capacity_constant_volume",
        "uc_cp": "unit_cell_heat_capacity_constant_pressure",
        "molar_cv": "molar_heat_capacity_constant_volume",
        "molar_cp": "molar_heat_capacity_constant_pressure"
    },
    "units_for_evaluation": {
        "uc_cv": "joule / kelvin",
        "uc_cp": "joule / kelvin",
        "molar_cv": "joule / kelvin / mol",
        "molar_cp": "joule / kelvin / mol"
    },
    "description": DESCRIPTION,
    "references": ["doi:10.1016/j.commatsci.2012.10.028"],
    "implemented_by": [
        "dmrdjenovich"
    ],
    "plug_in": plug_in
}
