def filter(materials):
    return True

def pre_filter(materials_list):
    if len(materials_list) != 2:
        return {'oxide': [], 'metal': []}
    metallic1 = {True: 0, False: 0}
    metallic2 = {True: 0, False: 0}
    for val in materials_list[0]._symbol_to_quantity['is_metallic']:
        metallic1[val._value._magnitude == 1] += 1
    for val in materials_list[1]._symbol_to_quantity['is_metallic']:
        metallic2[val._value._magnitude == 1] += 1
    metallic1 = metallic1[True] > metallic1[False]
    metallic2 = metallic2[True] > metallic2[False]
    if metallic1 and not metallic2:
        return {'metal': [materials_list[0]], 'oxide': [materials_list[1]]}
    if metallic2 and not metallic1:
        return {'metal': [materials_list[1]], 'oxide': [materials_list[0]]}
    return {'oxide': [], 'metal': []}

def plug_in(symbol_values):
    so = symbol_values['oxide.structure']
    sm = symbol_values['metal.structure']
    n = so.composition.reduced_composition.get(sm.composition.elements[0])

    m_oxide = float(so.composition.reduced_composition.weight)
    m_metal = float(sm.composition.reduced_composition.weight)

    result = m_oxide * so.density / (n * m_metal * sm.density)
    return {'pilling_bedworth_ratio': result, 'successful': True}


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