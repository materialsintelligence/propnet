from pymatgen.analysis.hhi.hhi import HHIModel

hhi_model = HHIModel()

def plug_in(symbol_values):
    formula = symbol_values['formula']
    hhi_reserve, hhi_production = hhi_model.get_hhi(formula)
    return {
        'hhi_reserve': hhi_reserve,
        'hhi_production': hhi_production
    }

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
    "description": "\nThe Herfindahl-Hirschman Index is a metric of how geographically dispersed elements in a chemical compound are.",
    "references": [
        "@article{Gaultois_2013,\n\tdoi = {10.1021/cm400893e},\n\turl = {https://doi.org/10.1021%2Fcm400893e},\n\tyear = 2013,\n\tmonth = {may},\n\tpublisher = {American Chemical Society ({ACS})},\n\tvolume = {25},\n\tnumber = {15},\n\tpages = {2911--2920},\n\tauthor = {Michael W. Gaultois and Taylor D. Sparks and Christopher K. H. Borg and Ram Seshadri and William D. Bonificio and David R. Clarke},\n\ttitle = {Data-Driven Review of Thermoelectric Materials: Performance and Resource Considerations},\n\tjournal = {Chemistry of Materials}\n}"
    ],
    "plug_in": plug_in
}
