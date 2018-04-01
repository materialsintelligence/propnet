import os

from habanero.cn import content_negotiation
from monty.serialization import loadfn, dumpfn

_REFERENCE_CACHE_PATH = os.path.join(os.path.dirname(__file__),'reference_cache.json'
_REFERENCE_CACHE = loadfn(_REFERENCE_CACHE_PATH)

def references_to_bib(refs):

    def url_to_bib(url):
        """
        Parses a url into a BibTeX-formatted referenced.

        Args:
            url: url string

        Returns:

        """

        return "@article\{,\n\turl = {}\n\}"

    parsed_refs = []
    for ref in refs:

        if ref in _REFERENCE_CACHE:
            parsed_ref = _REFERENCE_CACHE[ref]
        elif ref.startswith('url:'):
            url = ref.split('url:')[1]
            parsed_ref = url_to_bib(url)
        elif ref.startswith('doi:'):
            doi = ref.split('doi:')[1]
            parsed_ref = content_negotiation(doi, format='bibentry')
        else:
            raise ValueError('Unknown reference style for'
                             'reference: {}'.format(ref))

        if ref not in _REFERENCE_CACHE:
            _REFERENCE_CACHE[ref] = parsed_ref
            dumpfn(_REFERENCE_CACHE, _REFERENCE_CACHE_PATH)

        parsed_refs.append(ref)

    return refs