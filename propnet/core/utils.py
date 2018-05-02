import os

from uuid import uuid4, uuid5, NAMESPACE_URL

from habanero.cn import content_negotiation
from monty.serialization import loadfn, dumpfn


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


_REFERENCE_CACHE_PATH = os.path.join(os.path.dirname(__file__),'../data/reference_cache.json')
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