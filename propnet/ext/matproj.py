from datetime import datetime

from pymatgen import MPRester
from propnet.core.symbols import Property
from propnet.symbols import PropertyType, property_metadata

from propnet.core.materials import Material

mpr = MPRester()

# maps propnet property names to mp (mapidoc) keypath
MP_FROM_PROPNET_NAME_MAPPING = {
    'poisson_ratio': 'elasticity.poisson_ratio',
    'bulk_modulus': 'elasticity.K_Voigt_Reuss_Hill',
    'elastic_tensor_voigt': 'elasticity.elastic_tensor',
    'relative_permittivity': 'diel.poly_total',
    'density': 'density',
    'refractive_index': 'diel.n',
    'energy_above_hull': 'e_above_hull',
    'elastic_anisotropy': 'elasticity.elastic_anisotropy',
    'final_energy': 'final_energy',
    'final_energy_per_atom': 'final_energy_per_atom',
    'formation_energy_per_atom': 'formation_energy_per_atom',
    'piezoelectric_tensor': 'piezo.piezoelectric_tensor',
    'volume_unit_cell': 'volume',
    'structure': 'structure'
}
PROPNET_FROM_MP_NAME_MAPPING = {v: k for k, v in MP_FROM_PROPNET_NAME_MAPPING.items()}

# list of all available properties
AVAILABLE_MP_PROPERTIES = list(PROPNET_FROM_MP_NAME_MAPPING.keys()) + ['task_id']
PROPNET_PROPERTIES_ON_MP = list(MP_FROM_PROPNET_NAME_MAPPING.keys())


def import_all_props(mp_ids):
    """Given a list of materials ids, returns a list of all available properties
    for each material. One list per material id is stored in a list of lists.

    Args:
      mp_ids: materials ids for which properties are requested.

    Returns:
      list of lists containing all possible materials properties by mp_id

    """
    return import_props(mp_ids, AVAILABLE_MP_PROPERTIES)


def import_props(mp_ids, prop_list):
    """Given a list of materials ids and mp property names returns a list of property
    instances for each requested material. These lists are stored in an encapsulating
    list.

    Args:
      mp_ids: materials ids for which properties are requested.
      prop_list: requested MP properties for materials.

    Returns:
      list of lists containing requested materials properties by mp_id

    """
    query = mpr.query(criteria={"task_id": {'$in': mp_ids}}, properties=prop_list)
    print(query)
    # properties of all mp-ids inputted; list of lists
    properties = []
    for data in query:
        # properties of one mp-id
        mat_properties = []
        for key in data:
            if not data[key] is None:
                prop_type = property_metadata[PROPNET_FROM_MP_NAME_MAPPING[key]]
                p = Property(prop_type,
                             data[key],
                             'Imported from Materials Project',
                             [(data['task_id'], '')], {})
                mat_properties.append(p)
        properties.append(mat_properties)
    return properties


def materials_from_mp_ids(mp_ids):
    """

    Args:
        mp_ids:

    Returns:

    """
    query_results = mpr.query(criteria={'task_id': {'$in': mp_ids}},
                             properties=AVAILABLE_MP_PROPERTIES)
    materials = []
    for result in query_results:
        m = Material()
        for k, v in result.items():
            try:
                property_name = PROPNET_FROM_MP_NAME_MAPPING[k]
                property = Property(PropertyType[property_name],
                         v,
                         {
                             'source': 'Materials Project',
                             'uid': result['task_id'],
                             'date': datetime.now().strftime('%Y-%m-%d')
                         })
                m.add_property(property)
            except:
                print('Remove this try/except, for demo only ... some mappings incorrect')
        materials.append(m)
    return materials


def materials_from_formula(formula):
    query_results = mpr.query(criteria={'pretty_formula': formula},
                             properties=['task_id'])
    mpids = [entry['task_id'] for entry in query_results]
    return materials_from_mp_ids(mpids)

def mpid_from_formula(formula):
    query_results = mpr.query(criteria={'pretty_formula': formula},
                              properties=['task_id'])
    mpids = [entry['task_id'] for entry in query_results]
    if mpids:
        return mpids[0]
    else:
        return None