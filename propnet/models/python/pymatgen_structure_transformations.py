from pymatgen.transformations.standard_transformations import AutoOxiStateDecorationTransformation


def plug_in(self, symbol_values):
    s = symbol_values['s']
    trans = AutoOxiStateDecorationTransformation()
    s_oxi = trans.apply_transformation(s)
    return {
        's_oxi': s_oxi
    }

config = {
    "name": "pymatgen_structure_transformations",
    "connections": [
        {
            "inputs": [
                "s"
            ],
            "outputs": [
                "s_oxi"
            ]
        }
    ],
    "categories": [
        "pymatgen",
        "transformations"
    ],
    "symbol_property_map": {
        "s": "structure",
        "s_oxi": "structure_oxi"
    },
    "description": "\nThis model attempts to work out what oxidation state is on each crystallographic\nsite using the materials analysis code pymatgen.",
    "references": [
        "@article{Ong_2013,\n\tdoi = {10.1016/j.commatsci.2012.10.028},\n\turl = {https://doi.org/10.1016%2Fj.commatsci.2012.10.028},\n\tyear = 2013,\n\tmonth = {feb},\n\tpublisher = {Elsevier {BV}},\n\tvolume = {68},\n\tpages = {314--319},\n\tauthor = {Shyue Ping Ong and William Davidson Richards and Anubhav Jain and Geoffroy Hautier and Michael Kocher and Shreyas Cholia and Dan Gunter and Vincent L. Chevrier and Kristin A. Persson and Gerbrand Ceder},\n\ttitle = {Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis},\n\tjournal = {Computational Materials Science}\n}"
    ],
    "plug_in": plug_in
}