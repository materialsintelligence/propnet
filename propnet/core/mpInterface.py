from pymatgen import MPRester

def importProps(mpid = "mp-1234", propList = ["final_energy","density"]):
    m = MPRester("sNxknEySUTz2owRL")
    data = m.query(criteria={"task_id": mpid}, properties=propList)
    return(data)

a = importProps(mpid = "mp-1243" )[0]
props = []
#list of properties
for key in a:
    p = Property(key, a[key], )
    #make property
    props.append(p)
print(props)
