
def CompositeModel(Model):
    def __init__(self, connections, **kwargs):
        mat_inputs = connections['inputs']
        self.n_materials

        super(Model, self).__init__(**kwargs)

    def evaluate_constraints():
        pass

def plug_in(symbol_values):
    oxide_struct = symbol_values['oxide.structure']
    metal_struct = symbol_values['metal.structure']
    # Do something with these
    return {'pilling_bedworth_ratio': pbr}


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
                "oxide.structure",
                "metal.structure"
            ],
            "outputs": [
                "pilling_bedworth_ratio"
            ]
        }
    ],
    "categories": [
        "corrosion"
    ],
    "description": DESCRIPTION,
    "references": ["doi:10.1016/S0257-8972(02)00593-5"],
    "plug_in": plug_in
}