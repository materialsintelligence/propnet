import os
import sys
import logging
from io import StringIO
from monty.serialization import loadfn, dumpfn
from habanero.cn import content_negotiation
from propnet import print_logger, print_stream

from isbnlib import meta
from isbnlib.registry import bibformatters

from pybtex.database.input.bibtex import Parser
from pybtex.plugin import find_plugin
from pybtex.style.template import FieldIsMissing


_REFERENCE_CACHE_PATH = os.path.join(os.path.dirname(__file__),
                                     '../data/reference_cache.json')

if os.path.exists(_REFERENCE_CACHE_PATH):
    _REFERENCE_CACHE = loadfn(_REFERENCE_CACHE_PATH)
else:
    _REFERENCE_CACHE = dict()

logger = logging.getLogger(__name__)


def references_to_bib(refs, check_if_valid_citation=True):
    """
    Takes a list of reference strings and converts them to bibtex
    entries

    Args:
        refs ([str]): list of string references, which can be
            bibtex entries, digital object identifiers ("doi:DOI_GOES_HERE")
            or urls ("url:URL_GOES_HERE")
        check_if_valid_citation (bool): True checks to see if the
            reference generates a valid markdown-style citation. Throws ValueError
            if conversion is not successful.

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
        elif ref.startswith('isbn:'):
            isbn = ref.split('isbn:')[1]
            parsed_ref = bibformatters['bibtex'](meta(isbn))
        else:
            raise ValueError('Unknown reference style for '
                             'reference: {} (please either '
                             'supply a BibTeX string, or a string '
                             'starting with url: followed by a URL or '
                             'starting with doi: followed by a DOI)'.format(ref))

        if check_if_valid_citation:
            try:
                _ = references_to_markdown(parsed_ref)
            except Exception as ex:
                raise ValueError("Reference '{}' returned the following error.\n"
                                 "You may need to manually generate a bibtex string:\n"
                                 "{}".format(ref, ex))
        if ref not in _REFERENCE_CACHE:
            _REFERENCE_CACHE[ref] = parsed_ref
            dumpfn(_REFERENCE_CACHE, _REFERENCE_CACHE_PATH)
        parsed_refs.append(parsed_ref)
    return parsed_refs


def references_to_markdown(references):
    """Utility function to convert a BibTeX string containing
    references into a Markdown string.

    Args:
      references: BibTeX string

    Returns:
      Markdown string

    """

    pybtex_style = find_plugin('pybtex.style.formatting', 'plain')()
    pybtex_md_backend = find_plugin('pybtex.backends', 'markdown')
    pybtex_parser = Parser()

    # hack to not print labels (may remove this later)
    def write_entry(self, key, label, text):
        self.output(u'%s  \n' % text)
    pybtex_md_backend.write_entry = write_entry
    pybtex_md_backend = pybtex_md_backend()

    data = pybtex_parser.parse_stream(StringIO(references))
    data_formatted = pybtex_style.format_entries(data.entries.values())
    output = StringIO()
    try:
        pybtex_md_backend.write_to_stream(data_formatted, output)
    except FieldIsMissing as ex:
        raise ValueError("Reference is missing '{}' field".format(ex.field_name))

    # add blockquote style
    references_md = '> {}'.format(output.getvalue())
    references_md.replace('\n', '\n> ')

    return references_md


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

    def flush(self, *args, **kwargs):
        pass

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
    to an existing logging.Logger and write it to a string. Purpose is for debugging
    and verifying warning output in tests.

    Output is available while within the established context by using get_output(),
    when the LogSniffer object is stopped using stop(), or after stopping using get_output()

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
    def __init__(self, logger, level='INFO'):
        if not isinstance(logger, logging.Logger):
            raise ValueError("Need valid logger for sniffing")
        self._logger = logger
        self._sniffer = None
        self._sniffer_handler = None
        self._level = level
        self._old_logger_level = None
        self._last_output = None

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
            self._last_output = None
            self._sniffer = StringIO()
            self._sniffer_handler = logging.StreamHandler(stream=self._sniffer)
            self._sniffer_handler.setLevel(self._level)
            self._logger.addHandler(self._sniffer_handler)
            if self._logger.getEffectiveLevel() > self._sniffer_handler.level:
                self._old_logger_level = self._logger.getEffectiveLevel()
                self._logger.setLevel(self._level)

    def is_started(self):
        """
        Checks if the sniffer is actively sniffing.

        Returns: (bool) true if the sniffer is active, false otherwise

        """
        return self._sniffer is not None

    def stop(self, *args, **kwargs):
        """
        Stops recording messages passed to the logger, removes the sniffer,
        and returns the captured output. Returns None if sniffer is inactive.

        Keyword Args:
            replace_newline (str): a string with which to replace newline characters.
                default: '\n' (no replacement)
        Returns: (str) the output captured by the sniffer
            or None if sniffer is inactive

        """
        if self.is_started():
            output = self.get_output(*args, **kwargs)
            self._last_output = output
            self._logger.removeHandler(self._sniffer_handler)
            self._sniffer_handler = None
            self._sniffer.close()
            self._sniffer = None
            if self._old_logger_level is not None:
                self._logger.setLevel(self._old_logger_level)
                self._old_logger_level = None
        else:
            output = None
        return output

    def get_output(self, replace_newline='\n'):
        """
        Returns the output captured by the sniffer so far. Returns None if
        sniffer has never been started. If the sniffer had been started previously,
        but is currently stopped, returns the last output stored by the sniffer.

        Keyword Args:
            replace_newline (str): a string with which to replace newline characters.
                default: '\n' (no replacement)
        Returns: (str) the output captured by the sniffer,
            or None if sniffer has never been started

        """
        if self.is_started():
            return self._sniffer.getvalue().replace('\n', replace_newline)

        return self._last_output

    def clear(self):
        """
        Clears the captured output while keeping the sniffer active.

        """
        if self.is_started():
            self._sniffer.truncate(0)
            self._sniffer.seek(0)

# TODO: This should be exactly the same as maggma.utils.Timeout, but at the time of
#       writing, the package has not been updated on PyPI. Remove and replace usages
#       with version from maggma when it becomes available


if os.name == 'posix':
    # signal package is not supported by non-Unix systems, and so in turn, we cannot
    # timeout models on those platforms. There are few elegant solutions to killing threads in Python
    # that are cross-platform. If something better comes up, implement it.
    import signal

    class Timeout:
        # implementation courtesy of https://stackoverflow.com/a/22348885/637562

        def __init__(self, seconds=1, error_message=""):
            """
            Set a maximum running time for functions.

            Args:
                seconds (int): Seconds before TimeoutError raised, set to None to disable
                error_message (str): Error message to display with TimeoutError
            """

            self.seconds = int(seconds) if seconds else None
            self.error_message = error_message

        def _handle_timeout(self, signum, frame):
            raise TimeoutError(self.error_message)

        def __enter__(self):
            if self.seconds:
                signal.signal(signal.SIGALRM, self._handle_timeout)
                signal.alarm(self.seconds)

        def __exit__(self, type_, value, traceback):
            if self.seconds:
                signal.alarm(0)
else:
    class Timeout:
        def __init__(self, seconds=None, *args, **kwargs):
            if seconds is not None:
                logger.warning("Timeout class not implemented on non-Unix machines.")

        def __enter__(self):
            pass

        def __exit__(self, type_, value, traceback):
            pass
