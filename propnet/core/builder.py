from monty.json import jsanitize, MontyDecoder
from uncertainties import unumpy

from maggma.builders import Builder
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from propnet import logger
from propnet.core.quantity import create_quantity, StorageQuantity
from propnet.core.materials import Material
from propnet.core.graph import Graph
from propnet.core.provenance import ProvenanceElement
from propnet.models import DEFAULT_MODEL_DICT
from propnet.ext.matproj import MPRester
from pydash import get


class PropnetBuilder(Builder):
    """
    Basic builder for running propnet derivations on various properties
    """

    def __init__(self, materials, propstore, materials_symbol_map=None,
                 criteria=None, source_name="", **kwargs):
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
        if source_name == "":
            # Because this builder is not fully general, will keep this here
            self.source_name = "Materials Project"
        else:
            self.source_name = source_name
        super(PropnetBuilder, self).__init__(sources=[materials],
                                             targets=[propstore],
                                             **kwargs)

    def get_items(self):
        props = list(self.materials_symbol_map.keys())
        props += ["task_id", "pretty_formula", "run_type", "is_hubbard",
                  "pseudo_potential", "hubbards", "potcar_symbols", "oxide_type",
                  "final_energy", "unit_cell_formula", "created_at"]
        props = list(set(props))
        docs = self.materials.query(criteria=self.criteria, properties=props)
        self.total = docs.count()
        for doc in docs:
            logger.info("Processing %s", doc['task_id'])
            yield doc

    def process_item(self, item):
        # Define quantities corresponding to materials doc fields
        # Attach quantities to materials
        item = MontyDecoder().process_decoded(item)
        logger.info("Populating material for %s", item['task_id'])
        material = Material()

        if 'created_at' in item.keys():
            date_created = item['created_at']
        else:
            date_created = ""

        provenance = ProvenanceElement(source={"source": self.source_name,
                                               "source_key": item['task_id'],
                                               "date_created": date_created})

        for mkey, property_name in self.materials_symbol_map.items():
            value = get(item, mkey)
            if value:
                material.add_quantity(create_quantity(property_name, value,
                                                      provenance=provenance))

        # Add custom things, e. g. computed entry
        computed_entry = get_entry(item)
        material.add_quantity(create_quantity("computed_entry", computed_entry,
                                              provenance=provenance))
        material.add_quantity(create_quantity("external_identifier_mp", item['task_id'],
                                              provenance=provenance))

        input_quantities = material.get_quantities()

        # Use graph to generate expanded quantity pool
        logger.info("Evaluating graph for %s", item['task_id'])
        graph = Graph()
        graph.remove_models(
            {"dimensionality_cheon": DEFAULT_MODEL_DICT['dimensionality_cheon'],
             "dimensionality_gorai": DEFAULT_MODEL_DICT['dimensionality_gorai']})
        new_material = graph.evaluate(material)

        # Format document and return
        logger.info("Creating doc for %s", item['task_id'])
        # Gives the initial inputs that were used to derive properties of a
        # certain material.

        doc = {"inputs": [q for q in input_quantities]}
        for symbol, quantity in new_material.get_aggregated_quantities().items():
            all_qs = new_material._symbol_to_quantity[symbol]
            # Only add new quantities
            # TODO: Condition insufficiently general.
            #       Can end up with initial quantities added as "new quantities"
            if len(all_qs) == 1 and list(all_qs)[0] in input_quantities:
                continue
            # Write out all quantities as dicts including the
            # internal ID for provenance tracing
            qs = [q for q in all_qs]
            # THE listing of all Quantities of a given symbol.
            sub_doc = {"quantities": qs,
                       "mean": unumpy.nominal_values(quantity.value).tolist(),
                       "std_dev": unumpy.std_devs(quantity.value).tolist(),
                       "units": qs[0]['units'],
                       "title": quantity._symbol_type.display_names[0]}
            # Symbol Name -> Sub_Document, listing all Quantities of that type.
            doc[symbol.name] = sub_doc

        doc.update({"task_id": item["task_id"],
                    "pretty_formula": item["pretty_formula"]})
        return jsanitize(doc, strict=True)

    def update_targets(self, items):
        if not isinstance(items, list):
            raise TypeError("Expected list of items, "
                            "instead received {}".format(type(items)))
        self.propstore.update(items)


# This is a PITA, but right now there's no way to get this data from the
# built collection itself
def get_entry(doc):
    """
    Helper function to get a processed computed entry from the document

    Args:
        doc ({}): doc from which to get the entry

    Returns:
        (ComputedEntry) computed entry derived from doc

    """
    params = ["run_type", "is_hubbard", "pseudo_potential", "hubbards",
              "potcar_symbols", "oxide_type"]
    doc["potcar_symbols"] = ["%s %s" % (doc["pseudo_potential"]["functional"], l)
                             for l in doc["pseudo_potential"]["labels"]]
    entry = ComputedEntry(doc["unit_cell_formula"], doc["final_energy"],
                          parameters={k: doc[k] for k in params},
                          data={"oxide_type": doc['oxide_type']},
                          entry_id=doc["task_id"])
    entry = MaterialsProjectCompatibility().process_entries([entry])[0]
    return entry
