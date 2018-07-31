import os

from uuid import uuid4, uuid5, NAMESPACE_URL

from habanero.cn import content_negotiation
from monty.serialization import loadfn, dumpfn
from monty.json import jsanitize

from maggma.stores import MongoStore
from maggma.builder import Builder
from maggma.runner import Runner
from propnet import logger
from propnet.core.quantity import Quantity
from propnet.core.materials import Material
from propnet.core.graph import Graph
from pydash import get


NAMESPACE_PROPNET = uuid5(NAMESPACE_URL, "https://www.github.com/materialsintelligence/propnet/")


def uuid(name=None):
    """
    Returns a UUID, either deterministically (if a name is provided)
    or a random UUID (if no name provided).

    Args:
        name (str): any string, such as a model name or symbol name

    Returns: a UUID

    """
    if not name:
        return uuid4()  # uuid4 is random
    else:
        return uuid5(NAMESPACE_PROPNET, str(name))


_REFERENCE_CACHE_PATH = os.path.join(os.path.dirname(__file__), '../data/reference_cache.json')
_REFERENCE_CACHE = loadfn(_REFERENCE_CACHE_PATH)


def references_to_bib(refs):

    parsed_refs = []
    for ref in refs:

        if ref in _REFERENCE_CACHE:
            parsed_ref = _REFERENCE_CACHE[ref]
        elif ref.startswith('@'):
            parsed_ref = ref
        elif ref.startswith('url:'):
            # uses arbitrary key
            url = ref.split('url:')[1]
            parsed_ref = """@misc{{url:{0},
            url = {{{1}}}
            }}""".format(str(abs(url.__hash__()))[0:6], url)
        elif ref.startswith('doi:'):
            doi = ref.split('doi:')[1]
            parsed_ref = content_negotiation(doi, format='bibentry')
        else:
            raise ValueError('Unknown reference style for '
                             'reference: {} (please either '
                             'supply a BibTeX string, or a string '
                             'starting with url: followed by a URL or '
                             'starting with doi: followed by a DOI)'.format(ref))

        if ref not in _REFERENCE_CACHE:
            _REFERENCE_CACHE[ref] = parsed_ref
            dumpfn(_REFERENCE_CACHE, _REFERENCE_CACHE_PATH)

        print(parsed_ref)

        parsed_refs.append(parsed_ref)

    return parsed_refs


class PropnetBuilder(Builder):

    DEFAULT_MATERIAL_SYMBOL_MAP = {
    "structure": "structure",
    "elasticity.elastic_tensor": "elastic_tensor",
    "band_gap.search_gap.band_gap": "band_gap_pbe",
    }
    """
    Basic builder for running propnet derivations on various properties
    """
    def __init__(self, materials, propstore, materials_symbol_map=None,
                 criteria=None, **kwargs):
        """
        Args:
            materials (MongoStore): store of materials properties
            materials_symbol_map (dict): mapping of keys in materials
                store docs to symbols
            propstore (MongoStore): store of propnet properties
            **kwargs: kwargs for builder
        """
        self.materials = materials
        self.propstore = propstore
        self.criteria = criteria
        self.materials_symbol_map = materials_symbol_map \
                                    or self.DEFAULT_MATERIAL_SYMBOL_MAP

    def get_items(self):
        props = list(self.materials_symbol_map.keys())
        props += ["task_id", "pretty_formula"]
        docs = self.materials.query(criteria=self.criteria, properties=props)
        for doc in docs:
            logger.info("Processing %s", doc['task_id'])
            yield doc

    def process_item(self, item):
        # Define quantities corresponding to materials doc fields
        # Attach quantities to materials
        quantities = []
        material = Material()
        for mkey, property_name in self.materials_symbol_map.items():
            value = get(item, mkey)
            material.add_quantity(Quantity(property_name, value))

        # Use graph to generate expanded quantity pool
        graph = Graph()
        graph.add_material(material)
        graph.evaluate()

        # Format document and return
        doc = graph._symbol_to_quantity
        doc = {symbol.name: q_list for symbol, q_list in doc.items()}
        doc.update({"task_id": item["task_id"],
                    "pretty_formula": item["pretty_formula"]})
        return doc

    def update_targets(self, items):
        items = [jsanitize(item, strict=True) for item in items]
        self.propstore.update(items)

