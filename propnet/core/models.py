import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

# typing information, for type hinting only
from typing import *

from abc import ABCMeta, abstractmethod
from functools import wraps
from hashlib import sha256

from propnet.symbols import PropertyType
from propnet import logger
from propnet import ureg

from os.path import dirname
from ruamel.yaml import safe_load

# TODO: add pint integration
# TODO: decide on interface for conditions, assumptions etc.
# TODO: decide on interface for multiple-material models.


def load_metadata(path):

    with open(path, 'r') as f:
        metadata = f.read()

    # metadata file has yaml front matter with Markdown
    # as body of file (fairly standard format)
    metadata = metadata.split('---')

    markdown = metadata[2]
    metadata = safe_load(metadata[1])
    metadata['description'] = markdown

    return metadata


class AbstractModel(metaclass=ABCMeta):

    def __init__(self, metadata=None):

        if not metadata:
            try:
                # try loading from local file, see /models/ for examples
                path = '{}/../models/{}.yaml'.format(dirname(__file__), self.__class__.__name__)
                metadata = load_metadata(path)
            except Exception as e:
                print(e)
                metadata = {}

        self._metadata = metadata

        # retrieve units for each symbol
        self.unit_mapping = {}
        for symbol, name in self.symbol_mapping.items():
            try:
                self.unit_mapping[symbol] = {symbol: PropertyType[name].value.units
                                             for symbol, name in self.symbol_mapping.items()}
            except Exception as e:
                raise ValueError('Please check your property names in your symbol mapping, '
                                 'for property {} and model {}, are they all valid? '
                                 'Exception: {}'
                                 .format(name, self.__class__.__name__, e))

    @property
    def name(self):
        """

        Returns (str): Name of model

        """
        return self.__class__.__name__

    @property
    def title(self):
        """

        Returns (str): Title of model

        """
        return self._metadata.get('title', 'undefined')

    @property
    def tags(self):
        """

        Returns (list): List of tags (str)

        """
        return self._metadata.get('tags', [])

    @property
    def description(self):
        """

        Returns (str): Description of model as Markdown string

        """
        return self._metadata.get('description', '')

    @property
    def connections(self):
        """

        Each connection is a dictionary with keys 'input' and
        'output', with a list of symbols in each.

        Returns (list): List of connections

        """
        return self._metadata.get('connections', [])

    @property
    def symbol_mapping(self):
        """
        A mapping of a symbol named used within the model
        to the canonical symbol name, e.g. {"E": "youngs_modulus"}

        Returns (dict): symbol mapping dictionary

        """
        return self._metadata.get('symbol_mapping', {})

    @property
    def references(self):
        """
        References for a model. When defining a model, these
        should be given as a list of strings with either the
        prefix "url:" or "doi:", and a formatted BibTeX string
        will be generated

        Returns (list): list of BibTeX strings

        """

        refs = self._metadata.get('references', [])

        return refs

        # TODO: see below

        def parse_doi(doi):
            """
            Parses a DOI into a BibTeX-formatted referenced.

            Args:
                doi: DOI

            Returns: BibTeX string

            """

        def parse_url(url):
            """
            Parses a url into a BibTeX-formatted referenced.

            Args:
                url: url string

            Returns:

            """

        parsed_refs = []
        for ref in refs:
            if ref.startswith('url:'):
                url = ref.split('url:')[1]
                parsed_ref = parse_url(url)
            elif ref.startswith('doi:'):
                doi = ref.split('doi:')[1]
                parsed_ref = parse_url(doi)
            else:
                raise ValueError('Unknown reference style for'
                                 'model {}: {}'.format(self.name, ref))
            parsed_refs.append(ref)

        return refs

    @property
    def constraints(self):
        return self._metadata.get('equations', [])

    @property
    def constants(self):
        return self._metadata.get('constants', {})

    @property
    def equations(self):
        return self._metadata.get('equations', [])

    @property
    def input_symbols(self):
        return [d['inputs'] for d in self.connections]

    @property
    def output_symbols(self):
        return [d['outputs'] for d in self.connections]

    def _evaluate(self, symbol_values):
        """
        Evaluates a model directly.

        Args:
            symbol_values:

        Returns:

        """

        # TODO: rename this method?

        # check that we have equations defined
        if not self.equations:
            raise ValueError('Please implement the _evaluate '
                             'method for the {} model.'.format(self.name))

        eqns = [parse_expr(eq) for eq in self.equations]
        eqns = [eqn.subs(symbol_values) for eqn in eqns]

        possible_outputs = set()
        for eqn in eqns:
            possible_outputs = possible_outputs.union(eqn.free_symbols)

        outputs = {}
        for possible_output in possible_outputs:
            solutions = sp.nonlinsolve(eqns, possible_output)
            # taking first solution only, and only asking for one output symbol
            # so know length of output tuple for solutions will be 1
            solution = list(solutions)[0][0]
            if not isinstance(solution, sp.EmptySet):
                outputs[possible_output] = solution

        return outputs

    def evaluate(self, symbol_values):
        """
        Evaluate a model

        Args:
            symbol_values:

        Returns:

        """

        available_symbols = set(symbol_values.keys())

        # check we support this combination of inputs
        available_inputs = [len(set(possible_input_symbols) - available_symbols) > 0
                            for possible_input_symbols in self.input_symbols]
        if not any(available_inputs):
            raise ValueError("The {} model cannot generate any outputs for "
                             "these inputs: {}".format(self.name, available_symbols))

        # TODO: check our units
        # TODO: make this more robust

        try:
            # evaluate is allowed to fail
            out = self._evaluate(symbol_values)
            out['status'] = 'SUCCESS'
        except Exception as e:
            return {
                'status': 'FAILED',
                'message': str(e)
            }

        # add units to output

        return out

    def __hash__(self):
        """
        A unique model hash, SHA256 hash of the model class name.

        :return (str): 4-digit hex string
        """
        return sha256(self.__class__.__name__.encode('utf-8')).hexdigest()[0:4]

    @property
    def model_id(self):
        """
        A unique model identifier, function of model class name.

        :return (str): 4-digit hex string
        """
        return self.__hash__()