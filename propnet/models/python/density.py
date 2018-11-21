def plug_in(symbol_values):
    s = symbol_values['s']
    p = len(s.sites) / s.volume
    rho = float(s.density)
    mbar = rho / p
    v_a = 1 / p
    return {'p': len(s.sites) / s.volume,
            'rho': float(s.density),
            'v_a': v_a,
            'mbar': mbar}


DESCRIPTION = """
Model calculating the atomic density from the corresponding 
structure object of the material
"""

config = {
    "name": "density",
    "connections": [
        {
            "inputs": [
                "s"
            ],
            "outputs": [
                "p",
                "rho",
                "mbar",
                "v_a"
            ]
        }
    ],
    "categories": [
        "mechanical"
    ],
    "symbol_property_map": {
        "s": "structure",
        "p": "atomic_density",
        "rho": "density",
        "v_a": "volume_per_atom",
        "mbar": "mass_per_atom"
    },
    "description": DESCRIPTION,
    "references": [],
    "implemented_by": [
        "dmrdjenovich"
    ],
    "plug_in": plug_in
}
