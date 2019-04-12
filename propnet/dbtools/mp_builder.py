from monty.json import jsanitize, MontyDecoder
from uncertainties import unumpy

from maggma.builders import MapBuilder
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from propnet import logger
from propnet.core.quantity import QuantityFactory
from propnet.dbtools.storage import StorageQuantity
from propnet.core.materials import Material
from propnet.core.graph import Graph
from propnet.core.provenance import ProvenanceElement
from propnet.ext.matproj import MPRester
import pydash

# noinspection PyUnresolvedReferences
import propnet.models
# noinspection PyUnresolvedReferences
import propnet.symbols
from propnet.core.registry import Registry


class PropnetBuilder(MapBuilder):
    """
    Basic builder for running propnet derivations on various properties
    """

    def __init__(self, materials, propstore, materials_symbol_map=None,
                 criteria=None, source_name="", parallel=False,
                 max_workers=None, graph_timeout=None, **kwargs):
        """
        Args:
            materials (Store): store of materials properties
            materials_symbol_map (dict): mapping of keys in materials
                store docs to symbols
            propstore (Store): store of propnet properties
            criteria (dict): criteria for Mongodb find() query specifying
                criteria for records to process
            source_name (str): identifier for record source
            parallel (bool): True runs the graph algorithm in parallel with
                the number of workers specified by max_workers. Default: False (serial)
                Note: there will be no substantial speed-up from using a parallel
                runner with a parallel builder if there are long-running model evaluations
                that don't get timed out using the timeout keyword.
            max_workers (int): number of processes to spawn for parallel graph
                evaluation. Note that graph evaluation speed-up tops out at 3-4
                parallel processes. If the builder is run in a parallel maggma Runner,
                each will spawn max_workers number of processes to evaluate.
                For 4 parallel graph processes running on 3 parallel runners, this will spawn:
                1 main runner process + 3 parallel runners + (3 parallel
                runners * 4 graph processes) = 16 total processes
            graph_timeout (int): number of seconds after which to timeout per property
                (available only on Unix-based systems). Default: None (no limit)
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

        self.parallel = parallel
        if not parallel and max_workers is not None:
            raise ValueError("Cannot specify max_workers with parallel=False")
        self.max_workers = max_workers

        self.graph_timeout = graph_timeout

        self._graph_evaluator = Graph(parallel=parallel, max_workers=max_workers)

        props = list(self.materials_symbol_map.keys())
        props += ["task_id", "pretty_formula", "run_type", "is_hubbard",
                  "pseudo_potential", "hubbards", "potcar_symbols", "oxide_type",
                  "final_energy", "unit_cell_formula", "created_at"]
        props = list(set(props))

        super(PropnetBuilder, self).__init__(source=materials,
                                             target=propstore,
                                             ufn=self.process,
                                             projection=props,
                                             **kwargs)

    def process(self, item):
        # Define quantities corresponding to materials doc fields
        # Attach quantities to materials
        item = MontyDecoder().process_decoded(item)
        logger.info("Populating material for %s", item['task_id'])
        material = Material()

        if 'created_at' in item.keys():
            date_created = item['created_at']
        else:
            date_created = None

        provenance = ProvenanceElement(source={"source": self.source_name,
                                               "source_key": item['task_id'],
                                               "date_created": date_created})

        for mkey, property_name in self.materials_symbol_map.items():
            value = pydash.get(item, mkey)
            if value:
                material.add_quantity(
                    QuantityFactory.create_quantity(property_name, value,
                                                    units=Registry("units").get(property_name, None),
                                                    provenance=provenance))

        # Add custom things, e. g. computed entry
        computed_entry = get_entry(item)
        if computed_entry:
            material.add_quantity(QuantityFactory.create_quantity("computed_entry", computed_entry,
                                                                  provenance=provenance))
        else:
            logger.info(
                "Unable to create computed entry for {}".format(item['task_id']))
        material.add_quantity(QuantityFactory.create_quantity("external_identifier_mp", item['task_id'],
                                                              provenance=provenance))

        input_quantities = material.get_quantities()

        # Use graph to generate expanded quantity pool
        logger.info("Evaluating graph for %s", item['task_id'])

        new_material = self._graph_evaluator.evaluate(material, timeout=self.graph_timeout)

        # Format document and return
        logger.info("Creating doc for %s", item['task_id'])
        # Gives the initial inputs that were used to derive properties of a
        # certain material.

        doc = {"inputs": [StorageQuantity.from_quantity(q) for q in input_quantities]}
        for symbol, quantity in new_material.get_aggregated_quantities().items():
            all_qs = new_material._symbol_to_quantity[symbol]
            # If no new quantities of a given symbol were derived (i.e. if the initial
            # input quantity is the only one listed in the new material) then don't add
            # that quantity to the propnet entry document as a derived quantity.
            if len(all_qs) == 1 and list(all_qs)[0] in input_quantities:
                continue

            # Write out all quantities as dicts including the
            # internal ID for provenance tracing
            qs = [StorageQuantity.from_quantity(q).as_dict() for q in all_qs]
            # THE listing of all Quantities of a given symbol.
            sub_doc = {"quantities": qs,
                       "mean": unumpy.nominal_values(quantity.value).tolist(),
                       "std_dev": unumpy.std_devs(quantity.value).tolist(),
                       "units": quantity.units.format_babel() if quantity.units else None,
                       "title": quantity._symbol_type.display_names[0]}
            # Symbol Name -> Sub_Document, listing all Quantities of that type.
            doc[symbol.name] = sub_doc

        doc.update({"task_id": item["task_id"],
                    "pretty_formula": item.get("pretty_formula")})
        return jsanitize(doc, strict=True)


# This is a PITA, but right now there's no way to get this data from the
# built collection itself
# MPRester does build this entry for you, but requires a database query.
def get_entry(doc):
    """
    Helper function to get a processed computed entry from the document

    Args:
        doc ({}): doc from which to get the entry

    Returns:
        (ComputedEntry) computed entry derived from doc

    """

    required_fields = ["run_type", "is_hubbard", "pseudo_potential", "hubbards",
                       "oxide_type", "unit_cell_formula", "final_energy", "task_id",
                       "pseudo_potential.functional", "pseudo_potential.labels"]

    if any(not pydash.has(doc, field) for field in required_fields):
        return None

    params = ["run_type", "is_hubbard", "pseudo_potential", "hubbards",
              "potcar_symbols", "oxide_type"]

    doc["potcar_symbols"] = ["%s %s" % (doc["pseudo_potential"]["functional"], l)
                             for l in doc["pseudo_potential"]["labels"]]
    entry = ComputedEntry(doc["unit_cell_formula"], doc["final_energy"],
                          parameters={k: doc[k] for k in params},
                          data={"oxide_type": doc['oxide_type']},
                          entry_id=doc["task_id"])
    processed_entries = MaterialsProjectCompatibility().process_entries([entry])
    if processed_entries:
        return processed_entries[0]
    return None
