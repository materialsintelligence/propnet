from propnet.ext.matproj import MPRester
from propnet.core.tests.test_graph import TEST_DATA_DIR
from monty.json import jsanitize
import json
import os


def generate_composite_data_files():
    mpr = MPRester()
    mpids = ['mp-13', 'mp-24972']
    materials = mpr.get_materials_for_mpids(mpids)
    for m in materials:
        mpid = [q.value for q in m.get_quantities() if q.symbol == "external_identifier_mp"][0]
        with open(os.path.join(TEST_DATA_DIR, '{}.json'.format(mpid)), 'w') as f:
            qs = jsanitize(m.get_quantities(), strict=True)
            json.dump(qs, f)
