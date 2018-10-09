def plug_in(symbol_values):
    # placeholder, a little dumb
    # will be partly replaced with CrystalPrototypeClassifier
    structure = symbol_values['s']
    # support other anions too?
    if 'O' not in [sp.symbol for sp in structure.types_of_specie]:
        raise ValueError("No oxygen in structure")
    if structure.composition.anonymized_formula != 'ABC3':
        raise ValueError("Wrong anonymized formula")
    radii = []
    for sp in structure.types_of_specie:
        if sp.symbol != 'O':
            radii.append(float(sp.average_ionic_radius))
    return {
        'r_A': max(radii),
        'r_B': min(radii)
    }


DESCRIPTION = """
This model classifies whether a crystal is a perovskite, and returns information
on the A-site ionic radius and B-site ionic radius.",
"""

config = {
    "name": "perovskite_classifier",
    "connections": [
        {
            "inputs": [
                "s"
            ],
            "outputs": [
                "r_A",
                "r_B"
            ]
        }
    ],
    "categories": [
        "classifier"
    ],
    "symbol_property_map": {
        "r_A": "ionic_radius_a",
        "r_B": "ionic_radius_b",
        "s": "structure_oxi"
    },
    "description": DESCRIPTION,
    "references": [],
    "plug_in": plug_in
}
