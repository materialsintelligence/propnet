from pymatgen.analysis.hhi.hhi import HHIModel

hhi_model = HHIModel()

def plug_in(symbol_values):
    formula = symbol_values['formula']
    hhi_reserve, hhi_production = hhi_model.get_hhi(formula)
    if hhi_reserve is None:
        raise ValueError("No hhi_reserve")
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
    "display_names": ['HHI'],
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
    "variable_symbol_map": {
        "hhi_production": "hhi_production",
        "hhi_reserve": "hhi_reserve",
        "formula": "formula"
    },
    "description": DESCRIPTION,
    "references": ["doi:10.1021/cm400893e"],
    "implemented_by": [
        "mkhorton"
    ],
    "plug_in": plug_in
}
