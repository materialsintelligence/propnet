from propnet.core.models import AbstractModel

from os import environ
from monty.serialization import loadfn

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.util.provenance import StructureNL

class CrystalPrototypeClassifier(AbstractModel):

    # a library of crystal prototypes is not supplied with the Propnet distribution
    # this should be a .json file containing a list of dicts with key 'snl' for
    # a pymatgen StructureNL prototype and a key 'tags' with a dict of tags for
    # that prototype, e.g. 'tags': {'mineral': 'Perovskite'}
    path_to_prototypes = environ['PROPNET_PROTOTYPES']

    def _evaluate(self, symbol_values):

        prototypes = loadfn(self.path_to_prototypes)

        structure = symbol_values['structure']

        sm = StructureMatcher()
        ltol = 0.2
        stol = 0.3
        angle_tol = 5

        def match_prototype(structure):
            tags = []
            for d in prototypes:
                p = d['snl'].structure
                match = sm.fit_anonymous(p, s)
                if match:
                    tags.append(d['tags'])
            return tags

        def match_single_prototype(structure):
            sm.ltol = 0.2
            sm.stol = 0.3
            sm.angle_tol = 5
            tags = match_prototype(structure)
            while len(tags) > 1:
                sm.ltol *= 0.8
                sm.stol *= 0.8
                sm.angle_tol *= 0.8
                tags = match_prototype(structure)
                if sm.ltol < 0.01:
                    break
            return tags

        ###

        # thanks https://stackoverflow.com/a/30134039

        def partition(collection):
            if len(collection) == 1:
                yield [collection]
                return

            first = collection[0]
            for smaller in partition(collection[1:]):
                # insert `first` in each of the subpartition's subsets
                for n, subset in enumerate(smaller):
                    yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
                # put `first` in its own subset
                yield [[first]] + smaller

        something = list(range(1, 5))

        for n, p in enumerate(partition(something), 1):
            print(n, sorted(p))