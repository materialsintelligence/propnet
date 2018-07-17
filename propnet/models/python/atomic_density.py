from propnet import ureg

def evaluate_atomic_density(s):
    p = ureg.Quantity.from_tuple(s.num_sites / s.volume,
                                 [['angstroms', -3]])
    return {'p': p}

config = {
    "name": "Crystal structure density",
    "plug_in": evaluate_atomic_density,
    "categories": ["mechanical"],
    "references": [],
    "symbol_map": {"s": "structure",
                   "p": "atomic_density"},
    "connections": [{"inputs": ["s"],
                     "outputs": ["p"]}]
}
