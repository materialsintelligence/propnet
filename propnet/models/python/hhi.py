from pymatgen.analysis.hhi.hhi import HHIModel

hhi_model = HHIModel()

def plug_in(symbol_values):
    formula = symbol_values['formula']
    hhi_reserve, hhi_production = hhi_model.get_hhi(formula)
    return {
        'hhi_reserve': hhi_reserve,
        'hhi_production': hhi_production
    }


DESCRIPTION = """
The Herfindahl-Hirschman Index is a metric of how geographically 
dispersed elements in a chemical compound are.
"""

config = {
    "name": "hhi",
    "connections": [
        {
            "inputs": [
                "formula"
            ],
            "outputs": [
                "hhi_production",
                "hhi_reserve"
            ]
        }
    ],
    "categories": [
        "meta"
    ],
    "symbol_property_map": {
        "hhi_production": "hhi_production",
        "hhi_reserve": "hhi_reserve",
        "formula": "formula"
    },
    "description": DESCRIPTION,
    "references": ["doi:10.1021/cm400893e"],
    "plug_in": plug_in
}
