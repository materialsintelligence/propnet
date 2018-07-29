from pymatgen.analysis.cost.cost import CostDBElements, CostAnalyzer

cost_analyzer = CostAnalyzer(CostDBElements())

def plug_in(symbol_values):
    formula = symbol_values['formula']
    return {
        'cost_per_kg': cost_analyzer.get_cost_per_kg(formula),
        'cost_per_mol': cost_analyzer.get_cost_per_mol(formula)
    }

config = {
    "name": "cost",
    "connections": [
        {
            "inputs": [
                "formula"
            ],
            "outputs": [
                "cost_per_mol",
                "cost_per_kg"
            ]
        }
    ],
    "categories": [
        "meta"
    ],
    "symbol_property_map": {
        "cost_per_mol": "cost_per_mol",
        "cost_per_kg": "cost_per_kg",
        "formula": "formula"
    },
    "description": "\nA rough estimate of cost of a given material based on elemental prices, based on pymatgen's CostAnalyzer.",
    "references": [],
    "plug_in": plug_in
}
