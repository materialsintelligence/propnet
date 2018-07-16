from propnet import ureg

def evaluate_atomic_density(s):
    p = ureg.Quantity.from_tuple(s.num_sites / s.volume,
                                 [['angstroms', -3]])
    return {'p': p}

config = {
    "name": "Atomic density model",
    "function": evaluate_atomic_density,
    "tags": ["mechanical"],
    "references": [],
    "symbol_mapping": {"s": "structure",
                       "p": "atomic_density"},
    "connections": [{"inputs": ["s"],
                     "outputs": ["p"]}]
}
