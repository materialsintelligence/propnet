# Querying AFLOW needs to be done in manageable chunks
# These configuations define the chunks to download. Note that
# this will not download all properties for a material at once,
# so it must be run in its entirety to complete the material.

default_files_to_ingest = [
    'CONTCAR.relax.vasp',
    'AEL_elastic_tensor.json'
]

_lib3_filter_list = [[('auid', '__gt__', "aflow:{}".format(start))]
                     for start in (hex(i)[2:] for i in range(0, 16))]

default_query_configs = [
    {
        'catalog': 'icsd',
        'k': [200000],
        'exclude': [],
        'filter': [],
        'select': ['auid', 'aurl', 'compound'],
        'targets': ['data', 'auid']
    },
    {
        'catalog': 'icsd',
        'k': [200000],
        'exclude': ['compound', 'aurl'],
        'filter': [],
        'select': [],
        'targets': ['data']
    },
    {
        'catalog': 'icsd',
        'k': [10000],
        'exclude': ['compound'],
        'filter': [],
        'select': ['files', 'aurl'],
        'targets': ['data']
    },
    {
        'catalog': 'lib1',
        'k': [200000],
        'exclude': [],
        'filter': [],
        'select': ['auid', 'aurl', 'compound'],
        'targets': ['data', 'auid']
    },
    {
        'catalog': 'lib1',
        'k': [200000],
        'exclude': ['compound', 'aurl'],
        'filter': [],
        'select': [],
        'targets': ['data']
    },
    {
        'catalog': 'lib1',
        'k': [10000],
        'exclude': ['compound'],
        'filter': [],
        'select': ['files', 'aurl'],
        'targets': ['data']
    },
    {
        'catalog': 'lib2',
        'k': [200000],
        'exclude': [],
        'filter': [],
        'select': ['auid', 'aurl', 'compound'],
        'targets': ['data', 'auid']
    },
    {
        'catalog': 'lib2',
        'k': [200000],
        'exclude': ['compound', 'aurl'],
        'filter': [],
        'select': [],
        'targets': ['data']
    },
    {
        'catalog': 'lib2',
        'k': [10000],
        'exclude': ['compound'],
        'filter': [],
        'select': ['files', 'aurl'],
        'targets': ['data']
    },
    {
        'catalog': 'lib3',
        'k': [50000]*len(_lib3_filter_list),
        'exclude': [],
        'filter': _lib3_filter_list,
        'select': ['auid', 'aurl', 'compound'],
        'targets': ['data', 'auid']
    },
    {
        'catalog': 'lib3',
        'k': [50000]*len(_lib3_filter_list),
        'exclude': ['compound', 'aurl'],
        'filter': _lib3_filter_list,
        'select': [],
        'targets': ['data']
    },
    {
        'catalog': 'lib3',
        'k': [10000]*len(_lib3_filter_list),
        'exclude': ['compound'],
        'filter': _lib3_filter_list,
        'select': ['files', 'aurl'],
        'targets': ['data']
    },
]
