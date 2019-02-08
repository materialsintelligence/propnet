import os, sys, logging
from io import StringIO
from monty.serialization import loadfn, dumpfn
from habanero.cn import content_negotiation
from propnet import print_logger, print_stream

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
    This class provides a context manager to redirect stdout to a logger (propnet.print_logger).
    This way any print statements received from user-implemented functions
    can be automatically logged instead of being printed to the screen.

    Usage example:
        my_method()     # Statement whose output is not suppressed

        with PrintToLogger():
            foo()  # Some statement(s) which may produce screen output to suppress

        some_other_method()     # Statement whose output is not suppressed
        log = PrintToLogger.get_print_log() # Record anything printed by foo()
    """

    def __init__(self):
        self.logger = print_logger

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.info(line.rstrip())

    @staticmethod
    def get_print_log():
        """
        Gets contents of print log, containing any text printed to the screen while
        running under a PrintToLogger context.

        Returns: (str) contents of print log

        """
        return print_stream.getvalue()

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


class LogSniffer:
    """
    This class provides a context manager or explicit object to capture output written
    to an existing logging. Logger and write it to a string. Purpose is for debugging
    and verifying warning output in tests.

    Output is only available while within the established context by using get_output()
    or before the LogSniffer object is stopped with stop().

    Usage example:
        # Get logger whose messages you want to capture
        logger = logging.getLogger('my_logger_name')

        # Context manager usage
        with LogSniffer(logger) as ls:
            foo()  # Some statement(s) which write to the logger
            output = ls.get_output(replace_newline='') # Gets logger output and replaces newline characters
            expected_output = "Expected output"
            self.assertEqual(output, expected_output)

        # Explicit object usage
        ls = LogSniffer(logger)
        ls.start()      # Start capturing logger output
        foo()           # Statements which output to logger
        output = ls.stop()  # Get output and stop sniffer
        expected_output = "Expected output\n"
        self.assertEqual(output, expected_output)

    """
    def __init__(self, logger):
        if not isinstance(logger, logging.Logger):
            raise ValueError("Need valid logger for sniffing")
        self._logger = logger
        self._sniffer = None
        self._sniffer_handler = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __del__(self):
        self.stop()

    def start(self):
        """
        Starts recording messages passed to the logger. If already started, does nothing.

        """
        if not self.is_started():
            self._sniffer = StringIO()
            self._sniffer_handler = logging.StreamHandler(stream=self._sniffer)
            self._sniffer_handler.setLevel(logging.INFO)
            self._logger.addHandler(self._sniffer_handler)

    def is_started(self):
        """
        Checks if the sniffer is actively sniffing.

        Returns: (bool) true if the sniffer is active, false otherwise

        """
        return self._sniffer is not None

    def stop(self, **kwargs):
        """
        Stops recording messages passed to the logger, removes the sniffer,
        and returns the captured output. Returns None if sniffer is inactive.

        Keyword Args:
            replace_newline: a string with which to replace newline characters.
                default: '\n' (no replacement)
        Returns: (str) the output captured by the sniffer
            or None if sniffer is inactive

        """
        if self.is_started():
            output = self.get_output(**kwargs)
            self._logger.removeHandler(self._sniffer_handler)
            self._sniffer_handler = None
            self._sniffer.close()
            self._sniffer = None
        else:
            output = None
        return output

    def get_output(self, replace_newline='\n'):
        """
        Returns the output captured by the sniffer so far. Returns None if
        sniffer is not started.

        Keyword Args:
            replace_newline: a string with which to replace newline characters.
                default: '\n' (no replacement)
        Returns: (str) the output captured by the sniffer,
            or None if sniffer is inactive

        """
        if self.is_started():
            return self._sniffer.getvalue().replace('\n', replace_newline)

        return None

    def clear(self):
        """
        Clears the captured output while keeping the sniffer active.

        """
        if self.is_started():
            self._sniffer.truncate(0)
            self._sniffer.seek(0)
