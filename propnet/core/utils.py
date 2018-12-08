import os, sys
from monty.serialization import loadfn, dumpfn
from habanero.cn import content_negotiation
from propnet import logger

_REFERENCE_CACHE_PATH = os.path.join(os.path.dirname(__file__),
                                     '../data/reference_cache.json')
_REFERENCE_CACHE = loadfn(_REFERENCE_CACHE_PATH)

def references_to_bib(refs):
    """
    Takes a list of reference strings and converts them to bibtex
    entries

    Args:
        refs ([str]): list of string references, which can be
            bibtex entries, digital object identifiers ("doi:DOI_GOES_HERE")
            or urls ("url:URL_GOES_HERE")

    Returns:
        (list): list of bibtex formatted strings

    """
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
        parsed_refs.append(parsed_ref)
    return parsed_refs


class PrintToLogger:
    """
    This class provides a context manager to redirect stdout and stderr to the propnet
    info logger. This way any print statements received from user-implemented functions
    can be automatically logged instead of being printed to the screen.

    Usage example:
        my_method()     # Statement which will not suppress output

        with PrintToLogger():
            foo()  # Some statement(s) which may produce screen output to suppress

        some_other_method()     # Statement which will not suppress output
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = logger.info
        sys.stderr = logger.info

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
