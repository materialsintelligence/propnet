

def plug_in(self, symbol_values):
    structure = symbol_values['structure']
    output = {}
    for attr in self.symbol_mapping.keys():
        if hasattr(structure, attr):
            output[attr] = getattr(structure, attr, None)
    return output

config = {
    "name": "pymatgen_structure_properties",
    "connections": [
        {
            "inputs": [
                "structure"
            ],
            "outputs": [
                "num_sites",
                "volume"
            ]
        }
    ],
    "categories": [
        "pymatgen"
    ],
    "symbol_property_map": {
        "structure": "structure",
        "num_sites": "nsites",
        "volume": "volume_unit_cell",
        "composition": "composition"
    },
    "description": "\nProperties of a crystal structure, such as the number of sites in its unit cell and its space group,\nas calculated by pymatgen.",
    "references": [
        "@article{Ong_2013,\n\tdoi = {10.1016/j.commatsci.2012.10.028},\n\turl = {https://doi.org/10.1016%2Fj.commatsci.2012.10.028},\n\tyear = 2013,\n\tmonth = {feb},\n\tpublisher = {Elsevier {BV}},\n\tvolume = {68},\n\tpages = {314--319},\n\tauthor = {Shyue Ping Ong and William Davidson Richards and Anubhav Jain and Geoffroy Hautier and Michael Kocher and Shreyas Cholia and Dan Gunter and Vincent L. Chevrier and Kristin A. Persson and Gerbrand Ceder},\n\ttitle = {Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis},\n\tjournal = {Computational Materials Science}\n}"
    ],
    "plug_in": plug_in
}