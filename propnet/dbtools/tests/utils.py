from propnet.dbtools.tests.test_correlation import PROPNET_PROPS, TEST_DIR as CORR_TEST_DIR
from propnet.dbtools.tests.test_mp_builder import TEST_DIR as MP_TEST_DIR
from propnet.dbtools.separation import SeparationBuilder
from propnet.dbtools.mp_builder import PropnetBuilder
from maggma.advanced_stores import MongograntStore
from maggma.runner import Runner
from maggma.stores import MemoryStore
from monty.serialization import dumpfn
from monty.json import jsanitize
import json
import os


# Just here for reference, in case anyone wants to create a new set
# of test materials. Requires mongogrant read access to knowhere.lbl.gov.
def create_correlation_test_docs():
    """
    Creates JSON file containing a certain number of materials and their
    necessary properties to load the propnet store for correlation tests.

    """
    n_materials = 200
    pnstore = MongograntStore("ro:knowhere.lbl.gov/mp_core", "propnet")
    pnstore.connect()
    cursor = pnstore.query(
        criteria={'$and': [
            {'$or': [{p: {'$exists': True}},
                     {'inputs.symbol_type': p}]}
            for p in PROPNET_PROPS]},
        properties=['task_id', 'inputs'] +
                   [p + '.mean' for p in PROPNET_PROPS] +
                   [p + '.units' for p in PROPNET_PROPS] +
                   [p + '.quantities' for p in PROPNET_PROPS])
    data = []
    for item in cursor:
        if len(data) < n_materials:
            data.append(item)
        else:
            cursor.close()
            break
    dumpfn(data, os.path.join(CORR_TEST_DIR, "correlation_propnet_data.json"))


def create_correlation_quantity_indexed_docs():
    """
    Outputs JSON file containing the same data from create_correlation_test_docs()
    but as individual quantities. This mimics the quantity-indexed store.

    Must run create_correlation_test_docs() first and have the JSON file in the
    test directory.

    """
    pn_store = MemoryStore()
    q_store = MemoryStore()
    m_store = MemoryStore()
    with open(os.path.join(CORR_TEST_DIR, "correlation_propnet_data.json"), 'r') as f:
        data = json.load(f)
    pn_store.connect()
    pn_store.update(jsanitize(data, strict=True, allow_bson=True))
    sb = SeparationBuilder(pn_store, q_store, m_store)
    r = Runner([sb])
    r.run()
    q_data = list(q_store.query(criteria={}, properties={'_id': False}))
    dumpfn(q_data, os.path.join(CORR_TEST_DIR, "correlation_propnet_quantity_data.json"))


# Just here for reference, in case anyone wants to create a new set
# of test materials -jhm
def create_mp_builder_test_docs():
    """
    Create documents for propnet MP builder tests. Outputs JSON file to test directory.
    """
    formulas = ["BaNiO3", "Si", "Fe2O3", "Cs"]
    mgstore = MongograntStore("ro:matgen2.lbl.gov/mp_prod", "materials")
    builder = PropnetBuilder(
        mgstore, MemoryStore(), criteria={"pretty_formula": {"$in": formulas},
                                          "e_above_hull": 0})
    builder.connect()
    dumpfn(list(builder.get_items()), 
           os.path.join(MP_TEST_DIR, "test_materials.json"))
