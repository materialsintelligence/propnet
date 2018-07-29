from pymatgen.analysis.elasticity.elastic import ElasticTensor


def plug_in(self, symbol_values):
     tensor = ElasticTensor.from_voigt(symbol_values["C_ij"])
     structure = symbol_values["structure"]
     to_return = tensor.clarke_thermalcond(structure)
     if not isinstance(to_return, float):
         to_return = float(to_return)
     return {
       't': to_return
   }

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
    "description": "\nBased on the model posited in https://doi.org/10.1016/S0257-8972(02)00593-5,\npredicts the thermal conductivity of materials in the high temperature limit.\nMaterials have smaller values of Clarke thermal conductivity should be expected\nto have smaller thermal conductivity at high temperatures.\n\nIn particular, this model predicts that a material will have low thermal conductivity\nat high temperatures if it has the following broad characteristics\n\n1) A large molecular weight\n2) A complex crystal structure\n3) Non-directional bonding\n4) A large number of different atoms per molecule\n\nThis model currently uses the implementation in pymatgen.",
    "references": [
        "@article{Clarke_2003,\n\tdoi = {10.1016/s0257-8972(02)00593-5},\n\turl = {https://doi.org/10.1016%2Fs0257-8972%2802%2900593-5},\n\tyear = 2003,\n\tmonth = {jan},\n\tpublisher = {Elsevier {BV}},\n\tvolume = {163-164},\n\tpages = {67--74},\n\tauthor = {David R. Clarke},\n\ttitle = {Materials selection guidelines for low thermal conductivity thermal barrier coatings},\n\tjournal = {Surface and Coatings Technology}\n}"
    ],
    "plug_in": plug_in
}