from scipy.integrate import quad
from math import exp

def plug_in(symbol_values):
    t_d = symbol_values['debye_temperature']
    temp = symbol_values['temperature']
    t_ratio = temp / t_d
    integrand = lambda x: (x**4 * exp(x) / (exp(x) - 1)**2)
    c_v = 9 * 8.314 * t_ratio**3 * quad(integrand, 0, t_ratio**-1)[0]
    return {"molar_heat_capacity_constant_volume": c_v}


DESCRIPTION = """
Constant volume heat-capacity derived from the debye model
of solids
"""

config = {
    "name": "heat_capacity_debye",
    "connections": [
        {
            "inputs": [
                "debye_temperature",
                "temperature"
            ],
            "outputs": [
                "molar_heat_capacity_constant_volume",
            ]
        }
    ],
    "categories": [
        "thermal"
    ],
    "description": DESCRIPTION,
    "references": ["doi:10.1016/j.commatsci.2012.10.028"],
    "plug_in": plug_in
}