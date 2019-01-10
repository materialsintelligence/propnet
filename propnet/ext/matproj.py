from propnet.core.materials import Material
from propnet.core.quantity import Quantity
from propnet.core.provenance import ProvenanceElement

from pymatgen import MPRester as _MPRester


# TODO: Distinguish this from the MP rester proper
# TODO: do we really need the duplicate methods for lists/single material?
# TODO: a more or less universal query scheme
class MPRester(_MPRester):

    mapping = {
    "material_id": "external_identifier_mp",
    "band_gap.search_gap.band_gap": "band_gap_pbe",
    #"band_structure": "null_symbol",
    #"band_structure_uniform": "null_symbol",
    "computed_entry": "computed_entry",
    #"dos": "null_symbol",
    "diel.n": "refractive_index",
    "diel.poly_total": "relative_permittivity",
    #"diel.e_electronic": "null_symbol",
    "diel.e_total": "dielectric_tensor",
    #"diel.poly_electronic": "null_symbol",
    "diel.pot_ferroelectric": "potentially_ferroelectric",
    "pretty_formula": "formula",
    "e_above_hull": "energy_above_hull",
    "elasticity.elastic_tensor": "elastic_tensor_voigt",
    # "elasticity.G_Reuss": "shear_modulus",
    # "elasticity.G_VRH": "shear_modulus",
    # "elasticity.G_Voigt": "shear_modulus",
    # "elasticity.K_Reuss": "bulk_modulus",
    # "elasticity.K_VRH": "bulk_modulus",
    # "elasticity.K_Voigt": "bulk_modulus",
    # "elasticity.elastic_anisotropy": "elastic_anisotropy",
    # "elasticity.poisson_ratio": "poisson_ratio",
    "formation_energy_per_atom": "formation_energy_per_atom",
    "magnetic_type": "magnetic_order",
    "oxide_type": "oxide_type",
    "piezo.piezoelectric_tensor": "piezoelectric_tensor",
    #"piezo.v_max": "null_symbol", # TODO": "add property
    #"piezo.eij_max": "null_symbol", # TODO": "add property
    "structure": "structure",
    #"total_magnetization": "null_symbol",
        #  TODO": "add property total_magnetization_per_unit_cell
}

    def __init__(self, api_key=None):
        _MPRester.__init__(self, api_key)

    def get_mpid_from_formula(self, formula):
        """
        Returns a Materials Project ID from a formula, assuming
        the most stable structure for that formula.

        Args:
            formula (str): formula string

        Returns:
            mp-id string

        """
        q = self.query(criteria={'pretty_formula': formula},
                       properties=['material_id', 'e_above_hull'])
        # Sort so that most stable is first
        q = sorted(q, key=lambda x: x.get('e_above_hull'))
        if len(q) > 0:
            return q[0]['material_id']
        else:
            return None

    def get_properties_for_mpids(self, mpids, filter_null_properties=True):
        """
        Retrieve properties from the Materials Project
        for a given list of Materials Project IDs.

        Args:
            mpids ([str]): a list of Materials Project IDs

        Returns:
            ([Dict]) a list of property dictionaries

        """
        all_properties = list(self.mapping.keys())
        property_query = self.query(criteria={'material_id': {'$in': mpids}},
                                    properties=all_properties)

        q = {doc['material_id']: doc for doc in property_query}

        entry_query = self.get_entries({'material_id': {'$in': mpids}})
        computed_entries = {e.entry_id: e for e in entry_query}

        for mpid, doc in q.items():
            doc['computed_entry'] = computed_entries[mpid]
            if filter_null_properties:
                q[mpid] = {k: v for k, v in doc.items() if v is not None}

        return list(q.values())

    def get_properties_for_mpid(self, mpid, filter_null_properties=True):
        """
        A version of get_properties_for_mpids for a single
        mpid.

        Args:
            mpid (str): a Materials Project ID

        Returns:
            (Dict) a dictionary of property values keyed by property names

        """
        if len(self.get_properties_for_mpids([mpid])) > 0:
            return self.get_properties_for_mpids(
                [mpid], filter_null_properties=filter_null_properties)[0]
        else:
            return []

    def get_materials_for_mpids(self, mpids, filter_null_properties=True):
        """
        Retrieve a list of Materials from the materials
        Project for a given list of Materials Project IDs.

        Args:
            mpids: a list of Materials Project IDs

        Returns:

        """

        materials_properties = self.get_properties_for_mpids(
            mpids, filter_null_properties=filter_null_properties)
        materials = []

        for material_properties in materials_properties:
            material = Material()
            for property_name, property_value in material_properties.items():
                provenance = ProvenanceElement(source='Materials Project')
                quantity = Quantity(self.mapping[property_name], property_value,
                                    provenance=provenance)
                material.add_quantity(quantity)
            materials.append(material)

        return materials

    def get_material_for_mpid(self, mpid, filter_null_properties=True):
        """
        A version of get_materials for a single mpid.

        Args:
            mpid: a Materials Project ID

        Returns: a Material object

        """
        if len(self.get_materials_for_mpids([mpid])) > 0:
            return self.get_materials_for_mpids(
                [mpid], filter_null_properties=filter_null_properties)[0]
        else:
            return None
