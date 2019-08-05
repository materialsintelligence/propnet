from propnet.dbtools.tests.test_correlation import PROPNET_PROPS, \
    TEST_DATA_DIR as CORR_TEST_DIR
from propnet.dbtools.tests.test_mp_builder import TEST_DATA_DIR as MP_TEST_DIR
from propnet.dbtools.separation import SeparationBuilder
from propnet.dbtools.correlation import CorrelationBuilder
from propnet.dbtools.mp_builder import PropnetBuilder
from maggma.advanced_stores import MongograntStore
from maggma.runner import Runner
from maggma.stores import MemoryStore
from monty.serialization import dumpfn
from monty.json import jsanitize
import json
import os

# NOTE: When regenerating files, please make sure to visually inspect
# the data to make sure it is right.

# Just here for reference, in case anyone wants to create a new set
# of test materials. Requires mongogrant read access to knowhere.lbl.gov.
def create_correlation_test_docs():
    """
    Creates JSON file containing a certain number of materials and their
    necessary properties to load the propnet store for correlation tests.

    """
    n_materials = 200
    pnstore = MongograntStore("ro:mongodb03.nersc.gov/propnet", "propnet_july2019")
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


def output_new_correlation_values():
    """
    Prints new values for correlation tests when new files are generated.

    """
    correlation_builder = _get_correlation_values()
    bv_data = correlation_builder.correlation_store.query(
        criteria={'property_x': 'bulk_modulus',
                  'property_y': 'vickers_hardness'}
    )
    
    print("Bulk Modulus/Vickers Hardness")
    
    for item in bv_data:
        print("'{}': {},".format(item['correlation_func'],
                                 item['correlation']))

    vb_data = correlation_builder.correlation_store.query(
        criteria={'property_y': 'bulk_modulus',
                  'property_x': 'vickers_hardness'}
    )

    print("Vickers Hardness/Bulk Modulus")

    for item in vb_data:
        print("'{}': {},".format(item['correlation_func'],
                                 item['correlation']))
        
    print('linlsq correlation values')

    bg_ad = correlation_builder.correlation_store.query_one(
        criteria={'property_x': 'band_gap_pbe',
                  'property_y': 'atomic_density',
                  'correlation_func': 'linlsq'}
    )

    bm_vh = correlation_builder.correlation_store.query_one(
        criteria={'property_x': 'bulk_modulus',
                  'property_y': 'vickers_hardness',
                  'correlation_func': 'linlsq'}
    )
    
    print("[{}, {}]".format(bg_ad['correlation'], 
                            bm_vh['correlation']))
    

def create_new_expected_outfile():
    builder = _get_correlation_values()
    builder.write_correlation_data_file(
        os.path.join(CORR_TEST_DIR, 'correlation_outfile.json')
    )


def _get_correlation_values():
    full_propstore = MemoryStore()
    with open(os.path.join(CORR_TEST_DIR, "correlation_propnet_data.json"), 'r') as f:
        data = json.load(f)
    full_propstore.connect()
    full_propstore.update(jsanitize(data, strict=True, allow_bson=True))
    correlation_store = MemoryStore()
    builder = CorrelationBuilder(full_propstore, correlation_store,
                                 props=PROPNET_PROPS,
                                 funcs='all',
                                 from_quantity_db=False)
    runner = Runner([builder])
    runner.run()
    return builder

# Just here for reference, in case anyone wants to create a new set
# of test materials -jhm
def create_mp_builder_test_docs():
    """
    Create documents for propnet MP builder tests. Outputs JSON file to test directory.
    """
    formulas = ["BaNiO3", "Si", "Fe2O3", "Cs"]
    mgstore = MongograntStore("ro:knowhere.lbl.gov/mp_core", "materials")
    builder = PropnetBuilder(
        mgstore, MemoryStore(), criteria={"pretty_formula": {"$in": formulas},
                                          "e_above_hull": 0})
    builder.connect()

    materials = list(builder.get_items())
    deprecated_item = mgstore.query_one({'deprecated': True, 'sbxn': 'core'})

    # Create fake, non-core sandboxed material to represent proprietary info
    sandboxed_item = materials[0].copy()
    sandboxed_item.update(
        {'sbxn': ['test_sbx'],
         'task_id': 'mp-fakesbx'}
    )
    del sandboxed_item['_id']
    materials.extend([deprecated_item, sandboxed_item])
    dumpfn(materials, os.path.join(MP_TEST_DIR, "test_materials.json"))
