# Querying AFLOW needs to be done in manageable chunks
# These configuations define the chunks to download. Note that
# this will not download all properties for a material at once,
# so it must be run in its entirety to complete the material.

default_files_to_ingest = [
    'CONTCAR.relax.vasp',
    'AEL_elastic_tensor.json'
]
"""File names to download for each entry if they exist.
"""

_lib3_filter_list = [('auid', '__gt__', "aflow:{}".format(start))
                     for start in (hex(i)[2:] for i in range(0, 16))]
"""This set of filters divides the lib3 database up by AFLOW ID, sequentially
requesting IDs that start with "aflow:0", then "aflow:1", etc.
"""

default_query_configs = [
    {
        'catalog': 'icsd',
        'k': 200000,
        'exclude': [],
        'filter': [],
        'select': ['auid', 'aurl', 'compound'],
        'targets': ['data', 'auid']
    },
    {
        'catalog': 'icsd',
        'k': 200000,
        'exclude': ['compound', 'aurl'],
        'filter': [],
        'select': [],
        'targets': ['data']
    },
    {
        'catalog': 'icsd',
        'k': 10000,
        'exclude': ['compound'],
        'filter': [],
        'select': ['files', 'aurl'],
        'targets': ['data']
    },
    {
        'catalog': 'lib1',
        'k': 200000,
        'exclude': [],
        'filter': [],
        'select': ['auid', 'aurl', 'compound'],
        'targets': ['data', 'auid']
    },
    {
        'catalog': 'lib1',
        'k': 200000,
        'exclude': ['compound', 'aurl'],
        'filter': [],
        'select': [],
        'targets': ['data']
    },
    {
        'catalog': 'lib1',
        'k': 10000,
        'exclude': ['compound'],
        'filter': [],
        'select': ['files', 'aurl'],
        'targets': ['data']
    },
    {
        'catalog': 'lib2',
        'k': 200000,
        'exclude': [],
        'filter': [],
        'select': ['auid', 'aurl', 'compound'],
        'targets': ['data', 'auid']
    },
    {
        'catalog': 'lib2',
        'k': 200000,
        'exclude': ['compound', 'aurl'],
        'filter': [],
        'select': [],
        'targets': ['data']
    },
    {
        'catalog': 'lib2',
        'k': 10000,
        'exclude': ['compound'],
        'filter': [],
        'select': ['files', 'aurl'],
        'targets': ['data']
    }
]
"""Default set of query configurations for downloading the AFLOW database.
All entries must include the following keys in the format specified:

catalog (str): must specify the AFLOW catalog to query. Value can be None, but
    it is recommended that the catalog be specified because AFLUX queries tend to
    be more stable and take less time when catalog is specified.
exclude (list of str): list of keywords to explicity exclude from the query result.
    May be empty list.
targets (list of str): list of strings to specify which MongoDB stores will be updated
    with the data from the query. Must be 'data' and/or 'auid'.
select (list of str): list of keywords as strings to include in the query. Empty list means
    to request all possible keywords.
k (int): how many records to request with each HTTP request.
filter (list of tuples): tuples which represent an AFLOW filter specification 
    (see `aflow.control.Query` for more information). For example, to request materials with
    band gap < 1 and formula that contains Fe, filter would contain:
    `[('Egap', '__lt__', 1), ('compound', '__mod__', 'Fe')]`
    This is equivalent to the following using the AFLOW Python wrapper query:
    ```
    from aflow import K
    from aflow.control import search
    query = search().filter((K.Egap < 1) & (K.compound % 'Fe'))
    ```
"""

for lib3_filter in _lib3_filter_list:
    default_query_configs += [
        {
            'catalog': 'lib3',
            'k': 50000,
            'exclude': [],
            'filter': [lib3_filter],
            'select': ['auid', 'aurl', 'compound'],
            'targets': ['data', 'auid']
        },
        {
            'catalog': 'lib3',
            'k': 50000,
            'exclude': ['compound', 'aurl'],
            'filter': [lib3_filter],
            'select': [],
            'targets': ['data']
        },
        {
            'catalog': 'lib3',
            'k': 10000,
            'exclude': ['compound'],
            'filter': [lib3_filter],
            'select': ['files', 'aurl'],
            'targets': ['data']
        }
    ]

