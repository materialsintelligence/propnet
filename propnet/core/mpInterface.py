from pymatgen import MPRester
#from propnet.core.properties import Property



#maps propnet property names to mp (mapidoc) property 'location'
mpPropNames = {
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

#inverse of above dict
propnetPropNames = {v:k for k,v in mpPropNames.items()}

def importProps(mpid = ["mp-1234"], propList = ["final_energy","density"]):
    m = MPRester("sNxknEySUTz2owRL")
    query = m.query(criteria={"task_id": {'$in':mp_ids}}, properties=propList)
    print(query)
    #properties of all mp-ids inputted; list of lists
    properties = []
    for data in query:
        #properties of one mp-id
        matProperties = []
        for key in data:
            if data[key] != None:
                propType = property_metadata[propnetPropNames[key]];
                p = Property(propType, data[key], None, (data['task_id'], ''), None)
                p = 2
                matProperties.append(p)
        properties.append(matProperties)

    return(properties)




mpIDs = ["mp-1243","mp-1234"]
mpProplist = list(mpPropNames.values())
#always want task_id for source_id
if not 'task_id' in mpProplist:
    mpProplist.append('task_id')
importProps(mpIDs, mpProplist)
