
# TODO: filter and pre_filter, @dmrdjenovic
def filter(materials):
    pass

def pre_filter(materials_list):
    pass
    # return {'oxide': [material_1], "metal": [material_2]}

# TODO: Fill in how plug_in works, @dmrdjenovic
def plug_in(symbol_values):
    oxide_struct = symbol_values['oxide.structure']
    metal_struct = symbol_values['metal.structure']
    # Do something with these
    return {'pilling_bedworth_ratio': pbr}


DESCRIPTION = """
THIS IS THE DESCRIPTION OF THE PB RATIO
"""

config = {
    "name": "pilling_bedworth_ratio",
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
    "plug_in": plug_in,
    "filter": filter,
    "pre_filter": pre_filter

}