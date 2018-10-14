from pymatgen.analysis.elasticity.elastic import ElasticTensor


def plug_in(symbol_values):
    tensor = ElasticTensor.from_voigt(symbol_values["C_ij"])
    structure = symbol_values["structure"]
    to_return = tensor.clarke_thermalcond(structure)
    if not isinstance(to_return, float):
        to_return = float(to_return)
    return {'t': to_return}


DESCRIPTION = """
Based on the model posited in https://doi.org/10.1016/S0257-8972(02)00593-5,
predicts the thermal conductivity of materials in the high temperature limit.
Materials have smaller values of Clarke thermal conductivity should be expected
to have smaller thermal conductivity at high temperatures.

In particular, this model predicts that a material will have low thermal conductivity
at high temperatures if it has the following broad characteristics

1) A large molecular weight
2) A complex crystal structure
3) Non-directional bonding
4) A large number of different atoms per molecule

This model currently uses the implementation in pymatgen.
"""

config = {
    "name": "clarke_thermal_conductivity",
    "connections": [
        {
            "inputs": [
                "C_ij",
                "structure"
            ],
            "outputs": [
                "t"
            ]
        }
    ],
    "categories": [
        "thermal"
    ],
    "symbol_property_map": {
        "C_ij": "elastic_tensor_voigt",
        "structure": "structure",
        "t": "thermal_conductivity"
    },
    "description": DESCRIPTION,
    "references": ["doi:10.1016/S0257-8972(02)00593-5"],
    "plug_in": plug_in
}