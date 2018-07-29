from propnet import ureg

def evaluate_atomic_density(s):
    return {'p': ureg.Quantity.from_tuple(
        [len(s.sites) / s.volume, [['angstrom', -3]]]),
        'rho': float(s.density) * ureg.gram * ureg.centimeter ** -3}

config = {
    "name": "crystal_structure_density",
    "plug_in": evaluate_atomic_density,
    "categories": ["mechanical"],
    "references": [],
    "symbol_property_map": {
        "s": "structure",
        "p": "atomic_density",
        "rho": "density"},
    "connections": [{"inputs": ["s"],
                     "outputs": ["p", "rho"]}],
    "description": "Model calculating the atomic density from "
                   "the corresponding structure object of the material."
}
