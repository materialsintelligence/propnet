from pymatgen.analysis.find_dimension import find_dimension
from pymatgen.analysis.structure_analyzer import get_dimensionality


def plug_in(symbol_values):
    structure = symbol_values['structure']
    return {
        'dimensionality_cheon': find_dimension(structure),
        'dimensionality_gorai': get_dimensionality(structure)
    }

config = {
    "name": "dimensionality",
    "connections": [
        {
            "inputs": [
                "structure"
            ],
            "outputs": [
                "dimensionality_cheon",
                "dimensionality_gorai"
            ]
        }
    ],
    "categories": [
        "structure"
    ],
    "symbol_property_map": {
        "dimensionality_cheon": "dimensionality",
        "dimensionality_gorai": "dimensionality",
        "structure": "structure"
    },
    "description": "\nCalculates the dimensionality of a structure using one of two methods implemented in pymatgen.\n",
    "references": [
        "@article{Gorai_2016,\n\tdoi = {10.1039/c6ta04121c},\n\turl = {https://doi.org/10.1039%2Fc6ta04121c},\n\tyear = 2016,\n\tpublisher = {Royal Society of Chemistry ({RSC})},\n\tvolume = {4},\n\tnumber = {28},\n\tpages = {11110--11116},\n\tauthor = {Prashun Gorai and Eric S. Toberer and Vladan Stevanovi{\\'{c}}},\n\ttitle = {Computational identification of promising thermoelectric materials among known quasi-2D binary compounds},\n\tjournal = {Journal of Materials Chemistry A}\n}",
        "@article{Cheon_2017,\n\tdoi = {10.1021/acs.nanolett.6b05229},\n\turl = {https://doi.org/10.1021%2Facs.nanolett.6b05229},\n\tyear = 2017,\n\tmonth = {feb},\n\tpublisher = {American Chemical Society ({ACS})},\n\tvolume = {17},\n\tnumber = {3},\n\tpages = {1915--1923},\n\tauthor = {Gowoon Cheon and Karel-Alexander N. Duerloo and Austin D. Sendek and Chase Porter and Yuan Chen and Evan J. Reed},\n\ttitle = {Data Mining for New Two- and One-Dimensional Weakly Bonded Solids and Lattice-Commensurate Heterostructures},\n\tjournal = {Nano Letters}\n}"
    ],
    "plug_in": plug_in
}
