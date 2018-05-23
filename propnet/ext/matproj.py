from typing import Dict, List

from propnet.core.materials import Material
from propnet.core.quantity import Quantity

from pymatgen import MPRester as _MPRester


class MPRester(_MPRester):

    mapping = {
    "material_id": "external_identifier_mp",
    "band_gap": "band_gap_pbe",
    #"band_structure": "null_symbol",
    #"band_structure_uniform": "null_symbol",
    "computed_entry": "computed_entry",
    #"dos": "null_symbol",
    "diel.n": "refractive_index",
    "diel.poly_total": "relative_permittivity",
    #"diel.e_electronic": "null_symbol",
    #"diel.e_total": "null_symbol",
    #"diel.poly_electronic": "null_symbol",
    "diel.pot_ferroelectric": "potentially_ferroelectric",
    "pretty_formula": "formula",
    "e_above_hull": "energy_above_hull",
    "elasticity.elastic_tensor": "elastic_tensor_voigt",
    "elasticity.G_Reuss": "shear_modulus",
    "elasticity.G_VRH": "shear_modulus",
    "elasticity.G_Voigt": "shear_modulus",
    "elasticity.K_Reuss": "bulk_modulus",
    "elasticity.K_VRH": "bulk_modulus",
    "elasticity.K_Voigt": "bulk_modulus",
    "elasticity.elastic_anisotropy": "elastic_anisotropy",
    "elasticity.poisson_ratio": "poisson_ratio",
    "formation_energy_per_atom": "formation_energy_per_atom",
    "magnetic_type": "magnetic_order",
    "oxide_type": "oxide_type",
    "piezo.piezoelectric_tensor": "piezoelectric_tensor",
    #"piezo.v_max": "null_symbol", # TODO": "add property
    #"piezo.eij_max": "null_symbol", # TODO": "add property
    "structure": "structure",
    #"total_magnetization": "null_symbol", # TODO": "add property total_magnetization_per_unit_cell
}

    def __init__(self):
        super(MPRester, self).__init__()

    def get_mpid_from_formula(self, formula: str) -> str:
        """
        Returns a Materials Project ID from a formula, assuming
        the most stable structure for that formula.

        Args:
            formula: formula string

        Returns: mp-id string

        """
        q = self.query(criteria={'pretty_formula': formula},
                           properties=['material_id'])
        if len(q) > 0:
            return q[0]['material_id']
        else:
            return None

    def get_properties_for_mpids(self, mpids: List[str]) -> List[Dict]:
        """
        Retrieve properties from the Materials Project
        for a given list of Materials Project IDs.

        Args:
            mpids: a list of Materials Project IDs

        Returns: a list of property dictionaries

        """

        all_properties = list(self.mapping.keys())

        q = {doc['material_id']:doc for doc in
             self.query(criteria={'material_id': {'$in': mpids}}, properties=all_properties)}

        computed_entries = {e.entry_id:e for e in
                            self.get_entries({'material_id': {'$in': mpids}})}

        for mpid, doc in q.items():
            doc['computed_entry'] = computed_entries[mpid]

        return list(q.values())

    def get_properties_for_mpid(self, mpid: str) -> Dict:
        """
        A version of get_properties_for_mpids for a single
        mpid.

        Args:
            mpid: a Materials Project ID

        Returns: a property dictionary

        """
        if len(self.get_properties_for_mpids([mpid])) > 0:
            return self.get_properties_for_mpids([mpid])[0]
        else:
            return []

    def get_materials_for_mpids(self, mpids: List[str]) -> List[Material]:
        """
        Retrieve a list of Materials from the materials
        Project for a given list of Materials Project IDs.

        Args:
            mpids: a list of Materials Project IDs

        Returns:

        """

        materials_properties = self.get_properties_for_mpids(mpids)
        materials = []

        for material_properties in materials_properties:
            material = Material()
            for property_name, property_value in material_properties.items():
                quantity = Quantity(self.mapping[property_name], property_value)
                material.add_quantity(quantity)
            materials.append(material)

        return materials

    def get_material_for_mpid(self, mpid: str) -> Material:
        """
        A version of get_materials for a single mpid.

        Args:
            mpid: a Materials Project ID

        Returns: a Material object

        """
        if len(self.get_materials_for_mpids([mpid])) > 0:
            return self.get_materials_for_mpids([mpid])[0]
        else:
            return None