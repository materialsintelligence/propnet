from datetime import datetime

from pymatgen import MPRester
from pymatgen.core.structure import IStructure
from propnet.core.quantity import Quantity
from propnet.core.materials import Material
from propnet.symbols import DEFAULT_SYMBOLS

from propnet.core.materials import Material

# maps propnet symbol names to mp (mapidoc) keypath
MP_FROM_PROPNET_NAME_MAPPING = {
    'pretty_formula': 'pretty_formula',
    'elastic_tensor_voigt': 'elasticity.elastic_tensor',
    'elastic_anisotropy': 'elasticity.elastic_anisotropy',
    'relative_permittivity': 'diel.poly_total',
    'density': 'density',
    'refractive_index': 'diel.n',
    'energy_above_hull': 'e_above_hull',
    'final_energy': 'final_energy',
    'final_energy_per_atom': 'final_energy_per_atom',
    'formation_energy_per_atom': 'formation_energy_per_atom',
    'piezoelectric_tensor': 'piezo.piezoelectric_tensor',
    'volume_unit_cell': 'volume',
}
PROPNET_FROM_MP_NAME_MAPPING = {v: k for k, v in MP_FROM_PROPNET_NAME_MAPPING.items()}

# list of all available properties
AVAILABLE_MP_PROPERTIES = list(PROPNET_FROM_MP_NAME_MAPPING.keys()) + ['task_id', 'structure']
PROPNET_PROPERTIES_ON_MP = list(MP_FROM_PROPNET_NAME_MAPPING.keys())


def import_materials(mp_ids, api_key=None):
    """
    Given a list of material ids, returns a list of Material objects with all
    available properties from the Materials Project.
    Args:
        mp_ids (list<str>): list of material ids whose information will be retrieved.
        api_key (str): api key to be used to conduct the query.
    Returns:
        (list<Material>): list of material objects with associated data.
    """
    mpr = MPRester(api_key)
    to_return = []
    query = mpr.query(criteria={"task_id": {'$in': mp_ids}}, properties=AVAILABLE_MP_PROPERTIES)
    for data in query:
        # properties of one mp-id
        mat = Material()
        tag_string = data['task_id']
        if not data['structure'] is None:
            mat.add_quantity(Quantity('structure', data['structure'], [tag_string]))
            mat.add_quantity(Quantity('lattice_unit_cell', data['structure'].lattice.matrix, [tag_string]))
        for key in data:
            if not data[key] is None and key in PROPNET_FROM_MP_NAME_MAPPING.keys():
                prop_type = DEFAULT_SYMBOLS[PROPNET_FROM_MP_NAME_MAPPING[key]]
                p = Quantity(prop_type, data[key], [tag_string])
                mat.add_quantity(p)
        to_return.append(mat)
    return to_return


def import_material(mp_id, api_key=None):
    """
    Given a material id, returns a Material object with all available properties from
    the Materials Project
    Args:
        mp_id (str): material id whose information will be retrieved.
        api_key (str): api key to be used to conduct the query.
    Returns:
        (Material): material object with associated data.
    """
    return import_materials([mp_id], api_key)[0]


def materials_from_formula(formula, api_key=None):
    """
    Given a material chemical formula, returns all Material objects with a matching formula
    with all their available properties from the Materials Project.
    Args:
        formula (str): material's formula
        api_key (str): api key to be used to conduct the query.
    Returns:
        (list<Material>): all materials with matching formula
    """
    mpr = MPRester(api_key)
    query_results = mpr.query(criteria={'pretty_formula': formula},
                              properties=['task_id'])
    mpids = [entry['task_id'] for entry in query_results]
    return import_materials(mpids, api_key)
