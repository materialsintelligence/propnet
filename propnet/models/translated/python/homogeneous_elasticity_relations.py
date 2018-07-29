

def plug_in(self, symbol_values):
    """
    Performs the evaluation and computational logic to accommodate the homogeneous elasticity relations.
    1) Calculates the Elastic modulus and the Shear modulus if either is un-available.
    2) Calculates the remaining unknown properties based on the Elastic and Shear moduli.
    Args:
        symbol_values (dict): dictionary containing symbols mapped to floats.
    Returns:
        (dict): mapping from string symbol to float value giving result of applying the model to the
                given inputs.
    """
    outputs = dict()
    y = symbol_values.get("Y")
    g = symbol_values.get("G")
    if not y:
        y = HomogeneousElasticityRelations.calc_Y(symbol_values)
        outputs["Y"] = y
    if not g:
        g = HomogeneousElasticityRelations.calc_G(y, symbol_values)
        outputs["G"] = g
    remainder = HomogeneousElasticityRelations.get_constants_from_Y_G(y, g)
    for k in remainder.keys():
        if k not in symbol_values.keys():
            outputs[k] = remainder[k]
    return outputs

config = {
    "name": "homogeneous_elasticity_relations",
    "connections": [
        {
            "inputs": [
                "v",
                "G"
            ],
            "outputs": [
                "Y",
                "K",
                "l",
                "M"
            ]
        },
        {
            "inputs": [
                "v",
                "Y"
            ],
            "outputs": [
                "G",
                "K",
                "l",
                "M"
            ]
        },
        {
            "inputs": [
                "v",
                "K"
            ],
            "outputs": [
                "G",
                "Y",
                "l",
                "M"
            ]
        },
        {
            "inputs": [
                "v",
                "l"
            ],
            "outputs": [
                "G",
                "Y",
                "K",
                "M"
            ]
        },
        {
            "inputs": [
                "v",
                "M"
            ],
            "outputs": [
                "G",
                "Y",
                "K",
                "l"
            ]
        },
        {
            "inputs": [
                "G",
                "Y"
            ],
            "outputs": [
                "v",
                "K",
                "l",
                "M"
            ]
        },
        {
            "inputs": [
                "G",
                "K"
            ],
            "outputs": [
                "v",
                "Y",
                "l",
                "M"
            ]
        },
        {
            "inputs": [
                "G",
                "l"
            ],
            "outputs": [
                "v",
                "Y",
                "K",
                "M"
            ]
        },
        {
            "inputs": [
                "G",
                "M"
            ],
            "outputs": [
                "v",
                "Y",
                "K",
                "l"
            ]
        },
        {
            "inputs": [
                "Y",
                "K"
            ],
            "outputs": [
                "v",
                "G",
                "l",
                "M"
            ]
        },
        {
            "inputs": [
                "Y",
                "l"
            ],
            "outputs": [
                "v",
                "G",
                "K",
                "M"
            ]
        },
        {
            "inputs": [
                "Y",
                "M"
            ],
            "outputs": [
                "v",
                "G",
                "K",
                "l"
            ]
        },
        {
            "inputs": [
                "K",
                "l"
            ],
            "outputs": [
                "v",
                "G",
                "Y",
                "M"
            ]
        },
        {
            "inputs": [
                "K",
                "M"
            ],
            "outputs": [
                "v",
                "G",
                "Y",
                "l"
            ]
        },
        {
            "inputs": [
                "l",
                "M"
            ],
            "outputs": [
                "v",
                "G",
                "Y",
                "K"
            ]
        }
    ],
    "categories": [
        "mechanical"
    ],
    "symbol_property_map": {
        "v": "poisson_ratio",
        "G": "shear_modulus",
        "Y": "youngs_modulus",
        "K": "bulk_modulus",
        "l": "lame_first_parameter",
        "M": "p_wave_modulus"
    },
    "description": "\nModel inter-relating the various elastic quantities assuming a homogeneous medium.\nFrom any two elastic quantities the remainder can be derived.\n",
    "references": [
        "@misc{url:580346,\n            url = {https://en.wikipedia.org/wiki/Elastic_modulus}\n            }"
    ],
    "plug_in": plug_in
}