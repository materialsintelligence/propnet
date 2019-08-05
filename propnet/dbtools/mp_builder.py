from monty.json import jsanitize, MontyDecoder
from uncertainties import unumpy
from itertools import chain

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
from multiprocessing import current_process

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
                 criteria=None, source_name="", include_deprecated=False,
                 include_sandboxed=False, graph_parallel=False,
                 max_graph_workers=None, graph_timeout=None,
                 allow_child_process=False, **kwargs):
        """
        Args:
            materials (Store): store of materials properties
            materials_symbol_map (dict): mapping of keys in materials
                store docs to symbols
            propstore (Store): store of propnet properties
            criteria (dict): criteria for Mongodb find() query specifying
                criteria for records to process
            source_name (str): identifier for record source
            include_deprecated (bool): True processes materials marked as
                deprecated via the "deprecated" field. False skips those materials.
                If an entry does not have the "deprecated" field, it will be processed.
                Note that False will create a logical "and" with any criteria specified
                in "criteria". Default: False
            include_sandboxed (bool): True processes materials regardless of their MP
                sandbox. Note that False will create a logical "and" with any criteria specified
                in "criteria". False restricts materials to the "core" sandbox. Default: False
            graph_parallel (bool): True runs the graph algorithm in parallel with
                the number of workers specified by max_workers. Default: False (serial)
                Note: there will be no substantial speed-up from using a parallel
                runner with a parallel builder if there are long-running model evaluations
                that don't get timed out using the timeout keyword.
            max_graph_workers (int): number of processes to spawn for parallel graph
                evaluation. Note that graph evaluation speed-up tops out at 3-4
                parallel processes. If the builder is run in a parallel maggma Runner,
                each will spawn max_workers number of processes to evaluate.
                For 4 parallel graph processes running on 3 parallel runners, this will spawn:
                1 main runner process + 3 parallel runners + (3 parallel
                runners * 4 graph processes) = 16 total processes
            graph_timeout (int): number of seconds after which to timeout per property
                (available only on Unix-based systems). Default: None (no limit)
            allow_child_process (bool): If True, the user will be warned when graph_parallel
                is True and the builder is being run in a child process, usually
                indicating the builder is being run in a parallelized Runner, which is
                not recommended due to inefficiency in having to re-fork the graph processes
                with every new material. False suppresses this warning.
            **kwargs: kwargs for builder
        """
        self.materials = materials
        self.propstore = propstore

        self.include_deprecated = include_deprecated
        self.include_sandboxed = include_sandboxed

        filters = []
        if criteria:
            filters.append(criteria)

        if not include_deprecated:
            deprecated_filter = {
                "$or": [{"deprecated": {"$exists": False}},
                        {"deprecated": False}]
            }
            filters.append(deprecated_filter)

        if not include_sandboxed:
            sandboxed_filter = {'sbxn': 'core'}
            filters.append(sandboxed_filter)

        if len(filters) > 1:
            self.criteria = {'$and': filters}
        else:
            self.criteria = filters[0] if filters else None

        self.materials_symbol_map = materials_symbol_map \
                                    or MPRester.mapping
        if source_name == "":
            # Because this builder is not fully general, will keep this here
            self.source_name = "Materials Project"
        else:
            self.source_name = source_name

        self.graph_parallel = graph_parallel
        if not graph_parallel and max_graph_workers is not None:
            raise ValueError("Cannot specify max_workers with parallel=False")
        self.max_graph_workers = max_graph_workers

        self.graph_timeout = graph_timeout
        self.allow_child_process = allow_child_process
        self._graph_evaluator = Graph(parallel=graph_parallel, max_workers=max_graph_workers)

        props = list(self.materials_symbol_map.keys())
        props += ["task_id", "pretty_formula", "run_type", "is_hubbard",
                  "pseudo_potential", "hubbards", "potcar_symbols", "oxide_type",
                  "final_energy", "unit_cell_formula", "created_at", "deprecated", "sbxn"]
        props = list(set(props))

        super(PropnetBuilder, self).__init__(source=materials,
                                             target=propstore,
                                             query=self.criteria,
                                             ufn=self.process,
                                             projection=props,
                                             **kwargs)

    def process(self, item):
        if self.graph_parallel and not self.allow_child_process and \
                current_process().name != "MainProcess":
            logger.warning("It appears derive_quantities() is running "
                           "in a child process, possibly in a parallelized "
                           "Runner.\nThis is not recommended and will deteriorate "
                           "performance.")
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

        input_quantities = material.symbol_quantities_dict

        # Use graph to generate expanded quantity pool
        logger.info("Evaluating graph for %s", item['task_id'])

        new_material = self._graph_evaluator.evaluate(material, timeout=self.graph_timeout)

        # Format document and return
        logger.info("Creating doc for %s", item['task_id'])
        # Gives the initial inputs that were used to derive properties of a
        # certain material.

        doc = {"inputs": [StorageQuantity.from_quantity(q)
                          for q in chain.from_iterable(input_quantities.values())]}

        for symbol, quantities in new_material.symbol_quantities_dict.items():
            # If no new quantities of a given symbol were derived (i.e. if the initial
            # input quantity/ies is/are the only one/s listed in the new material) then don't add
            # that quantity to the propnet entry document as a derived quantity.
            if len(quantities) == len(input_quantities[symbol]):
                continue
            sub_doc = {}
            try:
                # Write out all quantities as dicts including the
                # internal ID for provenance tracing
                qs = [jsanitize(StorageQuantity.from_quantity(q), strict=True) for q in quantities]
            except AttributeError as ex:
                # Check to see if this is an error caused by an object
                # that is not JSON serializable
                msg = ex.args[0]
                if "object has no attribute 'as_dict'" in msg:
                    # Write error to db and logger
                    errmsg = "Quantity of Symbol '{}' is not ".format(symbol.name) + \
                        "JSON serializable. Cannot write quantities to database!"
                    logger.error(errmsg)
                    sub_doc['error'] = errmsg
                    qs = []
                else:
                    # If not, re-raise the error
                    raise ex
            sub_doc['quantities'] = qs
            doc[symbol.name] = sub_doc

        aggregated_quantities = new_material.get_aggregated_quantities()

        for symbol, quantity in aggregated_quantities.items():
            if symbol.name not in doc:
                # No new quantities were derived
                continue
            # Store mean and std dev for aggregated quantities
            sub_doc = {"mean": unumpy.nominal_values(quantity.value).tolist(),
                       "std_dev": unumpy.std_devs(quantity.value).tolist(),
                       "units": quantity.units.format_babel() if quantity.units else None,
                       "title": quantity.symbol.display_names[0]}
            # Symbol Name -> Sub_Document, listing all Quantities of that type.
            doc[symbol.name].update(sub_doc)

        doc.update({"task_id": item["task_id"],
                    "pretty_formula": item.get("pretty_formula"),
                    "deprecated": item.get("deprecated", False)})

        if self.include_sandboxed:
            doc.update({'sbxn': item.get("sbxn", [])})

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
