def plug_in(symbol_values):
    structure = symbol_values['structure']
    return {"num_sites": structure.num_sites,
            "volume": structure.volume,
            "composition": structure.composition,
            "lattice": structure.lattice,
            "atomic_density": len(structure.sites) / structure.volume,
            "density": float(structure.density)}


DESCRIPTION = """
Properties of a crystal structure, such as the number of sites in its 
unit cell and its space group, as calculated by pymatgen.
"""

config = {
    "name": "pymatgen_structure_properties",
    "connections": [
        {
            "inputs": [
                "structure"
            ],
            "outputs": [
                "num_sites",
                "volume",
                "composition",
                "lattice",
                "atomic_density",
                "density"
            ]
        }
    ],
    "categories": [
        "pymatgen"
    ],
    "symbol_property_map": {
        "structure": "structure",
        "num_sites": "nsites",
        "volume": "volume_unit_cell",
        "composition": "composition",
        "lattice": "lattice",
        "atomic_density": "atomic_density",
        "density": "density"
    },
    "description": DESCRIPTION,
    "references": ["doi:10.1016/j.commatsci.2012.10.028"],
    "implemented_by": [
        "mkhorton", "dmrdjenovich"
    ],
    "plug_in": plug_in
}
