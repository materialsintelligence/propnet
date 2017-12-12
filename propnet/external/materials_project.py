from pymatgen import MPRester
from propnet.core.properties import Property
from propnet.properties import PropertyType, property_metadata

# maps propnet property names to mp (mapidoc) property 'location'
mp_from_propnet_name_mapping = {
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
    'volume_unit_cell': 'volume'
}
propnet_from_mp_name_mapping = {v: k for k, v in mp_from_propnet_name_mapping.items()}
mp_prop_list = list(propnet_from_mp_name_mapping.keys()) + ['task_id']
propnet_prop_list = list(mp_from_propnet_name_mapping.keys())


def import_all_props(mp_ids):
    """Given a list of materials ids, returns a list of all available properties
    for each material. One list per material id is stored in a list of lists.

    Args:
      mp_ids: materials ids for which properties are requested.

    Returns:
      list of lists containing all possible materials properties by mp_id

    """
    return import_props(mp_ids, mp_prop_list)


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
    m = MPRester()
    query = m.query(criteria={"task_id": {'$in': mp_ids}}, properties=prop_list)
    print(query)
    # properties of all mp-ids inputted; list of lists
    properties = []
    for data in query:
        # properties of one mp-id
        mat_properties = []
        for key in data:
            if not data[key] is None:
                prop_type = property_metadata[propnet_from_mp_name_mapping[key]]
                p = Property(prop_type, data[key], 'Imported from Materials Project', [(data['task_id'], '')], {})
                mat_properties.append(p)
        properties.append(mat_properties)
    return properties
