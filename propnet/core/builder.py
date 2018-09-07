from monty.json import jsanitize
from monty.json import MontyDecoder
from uncertainties import unumpy

from maggma.builder import Builder
from propnet import logger
from propnet.core.quantity import Quantity
from propnet.core.materials import Material
from propnet.core.graph import Graph
from propnet.ext.matproj import MPRester
from pydash import get


class PropnetBuilder(Builder):
    """
    Basic builder for running propnet derivations on various properties
    """
    def __init__(self, materials, propstore, materials_symbol_map=None,
                 criteria=None, **kwargs):
        """
        Args:
            materials (Store): store of materials properties
            materials_symbol_map (dict): mapping of keys in materials
                store docs to symbols
            propstore (Store): store of propnet properties
            **kwargs: kwargs for builder
        """
        self.materials = materials
        self.propstore = propstore
        self.criteria = criteria
        self.materials_symbol_map = materials_symbol_map \
                                    or MPRester.mapping
        super(PropnetBuilder, self).__init__(sources=[materials],
                                             targets=[propstore],
                                             **kwargs)

    def get_items(self):
        props = list(self.materials_symbol_map.keys())
        props += ["task_id", "pretty_formula"]
        docs = self.materials.query(criteria=self.criteria, properties=props)
        self.total = docs.count()
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

        input_quantities = material.get_quantities()

        # Use graph to generate expanded quantity pool
        logger.info("Evaluating graph for %s", item['task_id'])
        graph = Graph()
        new_material = graph.evaluate(material)

        # Format document and return
        logger.info("Creating doc for %s", item['task_id'])
        doc = {"inputs": [quantity.as_dict() for quantity in input_quantities]}
        for symbol, quantity in new_material.get_aggregated_quantities().items():
            all_qs = new_material._symbol_to_quantity[symbol]
            # Only add new quantities
            if len(all_qs) == 1 and list(all_qs)[0] in input_quantities:
                continue
            qs = [quantity.as_dict() for quantity in all_qs]
            sub_doc = {"quantities": qs,
                       "mean": unumpy.nominal_values(quantity.value).tolist(),
                       "std_dev": unumpy.std_devs(quantity.value).tolist(),
                       "units": qs[0]['units'],
                       "title": quantity._symbol_type.display_names[0]}
            doc[symbol.name] = sub_doc
        doc.update({"task_id": item["task_id"],
                    "pretty_formula": item["pretty_formula"]})
        return jsanitize(doc, strict=True)

    def update_targets(self, items):
        items = [jsanitize(item, strict=True) for item in items]
        self.propstore.update(items)

