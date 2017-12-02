from pymatgen import MPRester
#from propnet.core.properties import Property



def importProps(mpid = ["mp-1234"], propList = ["final_energy","density"]):
    m = MPRester("sNxknEySUTz2owRL")
    data = m.query(criteria={"task_id": mpid}, properties=propList)
    return(data)

#maps propnet property names to mp (mapidoc) property 'location'
mppropname = {
    'poisson_ratio': 'elasticity.poisson_ratio',
    'bulk_modulus': 'elasticity.K_Voigt_Reuss_Hill',
    'elastic_tensor_voigt': 'elasticity.elastic_tensor',
    'relative_permittivity': 'diel.poly_total',
    'density': 'density',
    'refractive_index': 'diel.n',
    'energy_above_hull': 'e_above_hull'



}



mp_ids = ["mp-1243","mp-1234"]
proplist = list(mppropname.keys());

a = importProps(mpid = {'$in':mp_ids} )
props = []
#list of properties
print(a)
for key in a:
    p = Property(key, a[key], )
    #make property
    props.append(p)
print(props)
