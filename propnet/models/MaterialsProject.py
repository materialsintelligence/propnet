from typing import List, Dict

from propnet.core.models import AbstractModel
from propnet.core.materials import Material
from propnet.core.quantity import Quantity

from pymatgen import MPRester


class MaterialsProject(AbstractModel):

    # requires API key to be set via .pmgrc.yaml or PMG_MAPI_KEY env variable
    mpr = MPRester()

    def get_mpid_from_formula(self, formula: str) -> str:
        """
        Returns a Materials Project ID from a formula, assuming
        the most stable structure for that formula.

        Args:
            formula: formula string

        Returns: mp-id string

        """
        q = self.mpr.query(criteria={'pretty_formula': formula},
                           properties=['material_id'])
        return q[0]['material_id']

    def get_properties_for_mpids(self, mpids: List[str]) -> List[Dict]:
        """
        Retrieve properties from the Materials Project
        for a given list of Materials Project IDs.

        Args:
            mpids: a list of Materials Project IDs

        Returns: a list of property dictionaries

        """

        all_properties = set(self.symbol_mapping.keys())
        # some keys require specific API queries
        special_properties = {'band_structure', 'band_structure_uniform', 'dos', 'computed_entry'}
        all_properties = list(all_properties-special_properties)

        q = {d['material_id']:d for d in
             self.mpr.query(criteria={'material_id': {'$in': mpids}}, properties=all_properties)}

        computed_entries = {e.entry_id:e for e in
                            self.mpr.get_entries({'material_id': {'$in': mpids}})}

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
        return self.get_properties_for_mpids([mpid])[0]

    def get_materials(self, mpids: List[str]) -> List[Material]:
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
                quantity = Quantity(self.symbol_mapping[property_name], property_value)
                material.add_quantity(quantity)
            materials.append(material)

        return materials

    def get_material(self, mpid: str) -> Material:
        """
        A version of get_materials for a single mpid.

        Args:
            mpid: a Materials Project ID

        Returns: a Material object

        """
        return self.get_materials([mpid])[0]

    def plug_in(self, symbol_values):

        mpid = symbol_values.get('mpid',
                                 self.get_mpid_from_formula(symbol_values.get('formula')))

        properties = self.get_properties_for_mpid(mpid)

        return properties