from monty.json import jsanitize

from maggma.stores import MongoStore
from maggma.builder import Builder
from maggma.runner import Runner
from propnet import logger
from propnet.core.quantity import Quantity
from propnet.core.materials import Material
from propnet.core.graph import Graph
from pydash import get

from monty.serialization import dumpfn


class PropnetBuilder(Builder):

    DEFAULT_MATERIAL_SYMBOL_MAP = {
    "structure": "structure",
    "elasticity.elastic_tensor": "elastic_tensor_voigt",
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
        super(PropnetBuilder, self).__init__(sources=[materials],
                                             targets=[propstore])

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
        logger.info("Populating material for %s", item['task_id'])
        material = Material()
        for mkey, property_name in self.materials_symbol_map.items():
            value = get(item, mkey)
            if value:
                material.add_quantity(Quantity(property_name, value))

        # Use graph to generate expanded quantity pool
        logger.info("Evaluating graph for %s", item['task_id'])
        graph = Graph()
        graph.add_material(material)
        graph.evaluate()

        # Format document and return
        logger.info("Creating doc for %s", item['task_id'])
        doc = graph._symbol_to_quantity
        doc = {symbol.name: list(q_list) for symbol, q_list in doc.items()}
        doc.update({"task_id": item["task_id"],
                    "pretty_formula": item["pretty_formula"]})
        return jsanitize(doc, strict=True)

    def update_targets(self, items):
        items = [jsanitize(item, strict=True) for item in items]
        self.propstore.update(items)

