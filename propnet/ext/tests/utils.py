from monty.serialization import dumpfn
from propnet.ext.aflow import AflowAPIQuery
from propnet.dbtools.aflow_ingester import AflowIngester
import os
import json

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'test_data')


def create_aflow_test_docs():
    auids = [
        'aflow:0132ab6b9cddd429',  # Has an elastic tensor file
        'aflow:0136cbe39e59c471',  # An average joe material
        'aflow:d0c93a9396dc599e'
    ]

    query = AflowAPIQuery.from_pymongo({'auid': {'$in': auids}}, AflowIngester._available_kws, 50,
                                       property_reduction=True)
    if query.N != len(auids):
        auids_retrieved = [material['auid'] for page in query.responses.values()
                               for material in page.values()]
        auids_not_retrieved = set(auids) - set(auids_retrieved)
        raise ValueError("Not all materials retrieved. Perhaps they have been deprecated? "
                         "Unavailabie auids:\n{}".format(auids_not_retrieved))

    data = []
    for item in query:
        raw_data = item.raw
        try:
            contcar_data = item.files['CONTCAR.relax.vasp']()
        except Exception:
            contcar_data = None
        try:
            elastic_tensor_data = item.files['AEL_elastic_tensor.json']()
            elastic_tensor_data = json.loads(elastic_tensor_data)
        except Exception:
            elastic_tensor_data = None
        raw_data['CONTCAR_relax_vasp'] = contcar_data
        raw_data['AEL_elastic_tensor_json'] = elastic_tensor_data

        data.append(raw_data)

    dumpfn(data, os.path.join(TEST_DATA_DIR, 'aflow_store.json'))
