"""
Module containing classes and methods for Model functionality in Propnet code.
"""

# typing information, for type hinting only
from typing import *
from abc import abstractmethod

import numpy as np

from os.path import dirname, join, isfile

from ruamel.yaml import safe_load, safe_dump
from monty.serialization import loadfn

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from propnet.symbols import DEFAULT_SYMBOLS
from propnet import logger, ureg
from propnet.core.utils import uuid, references_to_bib
from propnet.core.exceptions import ModelEvaluationError, IncompleteData

def load_metadata(path):
    """
    Loads the .yaml file at the given path, returning a dictionary of .yaml contents.
    Custom model data at the end of the .yaml file is loaded in under the key "description".

    Args:
        path (str): valid file path to a .yaml file to be loaded.
    Returns:
        (dict<str,id>) representation of .yaml contents.
    """
    with open(path, 'r') as f:
        metadata = f.read()

    metadata = metadata.split('---')

    markdown = metadata[2]
    metadata = safe_load(metadata[1])
    metadata['description'] = markdown

    return metadata


class AbstractModel:
    """
    Baseclass for all models appearing in Propnet.

    Class contains a pointer to a metadata dictionary that stores all associated information about the model. Accessor
    methods are provided for accessing individual components of the dictionary. In general this dictionary should
    contain the following:
        (str) title -> (str) human-readable title for the model
        (str) tags -> (list<str>) list of categories applicable to the model.
        (str) references -> (list<str>) list of informational links explaining / supporting the model
        (str) symbol_mapping -> (dict<str,str>) keys are symbols used in equations of the model,
                                                values are Symbol enum values (Symbol.name field)
        (str) connections -> (list<dict<str,list<str>>>)
                                                Forms the list of outputs that can be generated from different sets of
                                                inputs. The outer list contains dictionaries. These dictionaries contain
                                                two keys: "inputs" and "outputs". Each key maps to a list of symbol
                                                strings that serve as the list of input / output property types required
                                                for evaluation by the model.
        (str) equations -> (list<str>) OPTIONAL, set of equations that establish the model.
                                       Evaluate method may be overridden in lieu of providing equations.
        (str) description -> (str) markdown-formatted text further describing / explaining the model.

    The following methods may be overridden for custom model behavior:
        constraints () -> dict<str,lambda(Quantity)->bool>
            Returns a dictionary mapping symbol to a lambda function that takes in a Quantity object and returns a bool
            indicating whether that Quantity meets all necessary conditions for validity.
        plug_in (dict<str,id>) -> dict<str,id>
            Given a dictionary specifying a value for a set of input symbols, returns the predicted value of the model
            for those inputs.

    Thus, in full generality, a model requires a .yaml file to specify appropriate parameters along with a .py file to
    specify any method overrides. Each model thus corresponds to two files with the same name, and an equivalently-named
    class in the .py file inheriting from AbstractModel. While this class is termed 'abstract', it contains no pure
    virtual methods (borrowing terms from C++) -- instead abstract is used to force the user to subclass.

    At runtime, all models' associated .py files are loaded in and a class object is created for each Model. These
    class objects' __init__ methods must be called to produce a Model object via instantiation. Upon instantiation,
    associated metadata for the models is loaded on demand (lazy loading) and the .yaml file metadata is read in only
    at this time. Such instantiation is performed automatically in

    Attributes:
        _metadata (dict<str,id>): stores the .yaml dictionary contents specified upon instantiation.
        unit_mapping (dict<str,Pint.unit>): mapping from symbols used in the model to their corresponding units.
    """

    def __init__(self, metadata=None, symbol_types=None, additional_symbols=None):
        """
        Constructs a Model object with the provided metadata.

        If the metadata is None, it attempts to load in the
        appropriate .yaml file at this time.
        Such a .yaml file must have a name equal to the class name.

        Args:
            metadata (dict<str,id>): metadata defining the model.
        """

        if additional_symbols:
            symbol_types = {symbol.name: symbol for symbol in symbol_types}
            DEFAULT_SYMBOLS.update(symbol_types)

        if symbol_types is None:
            symbol_types = DEFAULT_SYMBOLS

        if not metadata:
            try:
                # try loading from local file, see /models/ for examples
                path = '{}/../models/{}.yaml'.format(dirname(__file__), self.__class__.__name__)
                metadata = load_metadata(path)
            except Exception as e:
                logger.error(e)
                metadata = {}

        self._metadata = metadata

        # retrieve units for each symbol
        self.unit_mapping = {}
        for symbol, name in self.symbol_mapping.items():
            self.unit_mapping[symbol] = symbol_types[name].units

    # Suite of getter methods returning appropriate model data.

    @property
    def name(self) -> str:
        """

        Returns: Name of the model (this matches the class name),
        'title' gives a more human-readable title for the model.

        """
        return self.__class__.__name__

    @property
    def title(self) -> str:
        """

        Returns: A human-readable title for the model.

        """
        return self._metadata.get('title', 'undefined')

    @property
    def tags(self) -> List[str]:
        """

        Returns: A list of tags categories associated with the
        model.

        """
        return self._metadata.get('tags', [])

    @property
    def description(self) -> str:
        """

        Returns: A description of the model and how it works,
        provided as a Markdown-formatted string.

        """
        return self._metadata.get('description', '')

    @property
    def references(self):
        """
        References for a model. When defining a model, these should be given as a list of strings with either the
        prefix "url:" or "doi:", and a formatted BibTeX string will be generated

        Returns:
            (list<str>): list of BibTeX strings
        """

        refs = self._metadata.get('references', [])

        return references_to_bib(refs)

    @property
    def uuid(self):
        """
        A unique model identifier, function of model class name.
        """
        return uuid(self.__class__.__name__.encode('utf-8'))

    @property
    def model_id(self):
        return self.uuid

    @property
    def symbol_mapping(self) -> Dict[str, str]:
        """
        A mapping of a symbol named used within the model to the canonical symbol name, e.g. {"E": "youngs_modulus"}
        keys are symbols used in the model; values are Symbol enum values (Symbol.name field)

        Returns:
            (dict<str,str>): symbol mapping dictionary
        """
        return self._metadata.get('symbol_mapping', {})

    @property
    def connections(self):
        """
        Forms the list of outputs that can be generated from different sets of inputs. The outer list contains
        dictionaries. These dictionaries contain two keys: "inputs" and "outputs". Each key maps to a list of symbol
        strings that serve as the list of input / output property types required for evaluation by the model.

        Returns:
             (list<dict<str,list<str>>>): List of connections
        """
        return self._metadata.get('connections', [])

    @property
    def type_connections(self):
        """
        Froms the list of outputs that can be generated from different sets of inputs. The outer list contains
        dictionaries. These dictionaries contain two keys: "inputs" and "outputs". Each key maps to a list of
        Symbol_type strings that serve as the list of input / output propperty types required for evaluation.

        Returns:
            (list<dict<str,list<str>>>): List of connections
        """
        to_return = []
        for d in self.connections:
            building = dict()
            building['inputs'] = [self.symbol_mapping[x] for x in d['inputs']]
            building['outputs'] = [self.symbol_mapping[x] for x in d['outputs']]
            to_return.append(building)
        return to_return

    def gen_evaluation_lists(self):
        """
        Convenience method.
        Returns (list<(list<str>, list<str>)>))
                gives corresponding sets of model input symbols and input Symbol objects.
        """
        to_return = []
        for d in self.connections:
            l = d['inputs'] + self.constraint_symbols
            l_types = [self.symbol_mapping[x] for x in l]
            to_return.append((l, l_types))
        return to_return


    @property
    def input_symbols(self):
        """
        Returns:
            (list<str>): all sets of input symbols for the model
        """
        return [d['inputs'] for d in self.connections]

    @property
    def input_symbol_types(self):
        """
        Returns:
            (set<str>): all sets of input Symbol objects for the model
        """
        return {self.symbol_mapping[x] for x in self.input_symbols}

    @property
    def output_symbols(self):
        """
        Returns:
            (list<str>): all sets of output symbols for the model
        """
        return [d['outputs'] for d in self.connections]

    @property
    def output_symbol_types(self):
        """
        Returns:
            (list<str>): all sets of output Symbol objects for the model
        """
        to_return = set()
        for l in self.output_symbols:
            for i in l:
                to_return.add(self.symbol_mapping[i])
        return to_return

    @property
    def constraint_symbols(self):
        """
        Returns a list of symbols.
        These symbols are those whose value needs to be evaluated to determine if the model can be evaluated under the
        current conditions.
        Returns: ([str])
        """
        return []

    def type_constraint_symbols(self):
        """
        Returns the Symbols required for evaluation of constraints.
        Returns: ([Symbol])
        """
        to_return = list()
        for s in self.constraint_symbols:
            to_return.append(self.symbol_mapping[s])
        return to_return

    def check_constraints(self, constraint_inputs):
        """
        Returns a dictionary mapping symbol to a lambda function that takes in a Quantity object and returns a bool
        indicating whether that Quantity meets all necessary conditions for validity.

        Args:
            constraint_inputs (dict<str, float>): Mapping from string symbol to symbol value
        Returns:
            (bool): bool stating whether the constraints of the model are met.
        """
        return True



    @property
    def equations(self):
        """
        Returns:
            (list<str>): equations that define the model
        """
        return self._metadata.get('equations', [])

    def evaluate(self, symbol_values):
        """
        Given a set of symbol_values, performs error checking to see if the input symbol_values represents a valid input
        set based on the self.connections() method. If so, it returns a dictionary representing the value of plug_in
        applied to the inputs. The dictionary contains a "successful" key representing if plug_in was successful.

        Args:
            symbol_values (dict<str,pint.Quantity>): Mapping from string symbol to float value, giving inputs.
        Returns:
            (dict<str,pint.Quantity>), mapping from string symbol to float value giving result of applying the model to the
                               given inputs. Additionally contains a "successful" key -> bool pair.
        """

        # strip units from input
        for symbol, value in symbol_values.items():
            if isinstance(value, ureg.Quantity):
                converted = value.to(self.unit_mapping[symbol])
                symbol_values[symbol] = converted.magnitude

        available_symbols = set(symbol_values.keys())

        # check we support this combination of inputs
        available_inputs = [len(set(possible_input_symbols) - available_symbols) == 0
                            for possible_input_symbols in self.input_symbols]
        if not any(available_inputs):
            return {
                'successful': False,
                'message': "The {} model cannot generate any outputs for these inputs: {}".format(
                    self.name, available_symbols)
            }
        try:
            # evaluate is allowed to fail
            out = self.plug_in(symbol_values)
            out['successful'] = True
        except Exception as e:
            return {
                'successful': False,
                'message': str(e)
            }

        # add units to output
        for key in out:
            if key == 'successful':
                continue
            out[key] = ureg.Quantity(out[key], self.unit_mapping[key])
        return out

    def plug_in(self, symbol_values):
        """
        Given a set of symbol_values, plugs the values into the model and returns a dictionary of outputs representing
        the result of plugging in the symbol_values. symbol_values must contain a valid set of inputs as indicated in
        the connections method.

        Args:
            symbol_values (dict<str,float>): Mapping from string symbol to float value, giving inputs.
        Returns:
            (dict<str,float>) mapping from string symbol to float value giving result of applying the model to the
                              given inputs.
        """
        # Define sympy equations for the model
        if not self.equations:
            raise ValueError('Please implement the _evaluate '
                             'method for the {} model.'.format(self.name))
        eqns = [parse_expr(eq) for eq in self.equations]
        eqns = [eqn.subs(symbol_values) for eqn in eqns]
        # Generate outputs from the sympy equations.
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
                outputs[str(possible_output)] = float(sp.N(solution))
        return outputs


    def __hash__(self):
        return self.uuid.__hash__()

    def __eq__(self, other):
        return self.model_id == getattr(other, "model_id", None)

    def __repr__(self):
        return self.name

    def __str__(self):
        return "{} [{}]".format(self._metadata['title'], self.model_id)

    @property
    def test_data(self):

        test_file = join(dirname(__file__), '../models/test_data/{}.json'
                         .format(self.__class__.__name__))
        if not isfile(test_file):
            logger.warn(IncompleteData("Test data file is missing for {}".format(self.name)))
            return None
        else:
            return loadfn(test_file)

    # TODO: rename to test_model
    def test(self, test_data=None):
        """

        Args:
            test_data: list of test data

        Returns: False if tests fail or no test data supplied,
        True if tests pass.

        """

        if not test_data:
            test_data = self.test_data

        if test_data:

            for d in test_data:
                try:
                    model_outputs = self.evaluate(d['inputs'])
                    for k, known_output in d['outputs'].items():
                        model_output = model_outputs[k]
                        # TODO: remove, here temporarily
                        if hasattr(model_output, 'magnitude'):
                            model_output = model_output.magnitude
                        if (not isinstance(known_output, float)) and \
                                (not isinstance(known_output, list)):
                            if model_output != known_output:
                                raise ModelEvaluationError(
                                    "Model output does not match known output "
                                    "for {}".format(self.name))
                        elif not np.allclose(model_output, known_output):
                            raise ModelEvaluationError("Model output does not match known output "
                                                       "for {}".format(self.name))
                except Exception as e:
                    raise ModelEvaluationError("Failed testing: " + self.title + ": " + str(e))

            return True

        else:

            return False

    def to_yaml(self):

        data = {
            "title": self.name,
            "tags": self.tags,
            "references": self._metadata.get('references', []),
            "symbol_mapping": self.symbol_mapping,
            "connections": self.connections
        }

        if self.equations:
            data["equations"] = self.equations

        header = safe_dump(data)

        return "{}---\n{}".format(header, self.description)

    @property
    def _example_code(self):
        """
        Generates example code from test data, useful for
        documentation.

        Returns: example code for this model

        """

        if not self.test_data:
            return None

        example_inputs = self.test_data[0]['inputs']
        example_outputs = str(self.test_data[0]['outputs'])

        symbol_definitions = []
        evaluate_args = []
        for input_name, input_value in example_inputs.items():

            symbol_str = "{input_name} = {input_value}  # {symbol_name} in {units}".format(
                input_name=input_name,
                input_value=input_value,
                symbol_name=self.symbol_mapping[input_name],
                units=self.unit_mapping[input_name]
            )
            symbol_definitions.append(symbol_str)

            evaluate_str = "\t'{}': {}".format(input_name, input_name)
            evaluate_args.append(evaluate_str)

        symbol_definitions = '\n'.join(symbol_definitions)
        evaluate_args = '\n'.join(evaluate_args)

        example_code = """\
from propnet.models import {model_name}

{symbol_definitions}

model = {model_name}()
model.evaluate({{
{evaluate_args}
}})  # returns {example_outputs}
""".format(model_name=self.name,
           symbol_definitions=symbol_definitions,
           evaluate_args=evaluate_args,
           example_outputs=example_outputs
           )

        return example_code

class AbstractSuperModel(AbstractModel):
    """
    Extension of AbstractModel to handle additional fields in the case of evaluating a model
    with multiple Materials as an input to that model.

    Contains fields of AbstractModel, plus the following modifications:
        (str) material_mapping -> (dict<str,Material>) Material objects coupled as input to the SuperModel
        (str) symbol_mapping -> (dict<str,(str,{str})>) keys are symbols used in equations of the model,
                                                        values are a tuple with
                                                          1) Symbol enum values (Symbol.name field)
                                                          2) set of labels corresponding to a Material object labels,
                                                             dictating from which materials the symbol's value can be
                                                             drawn.
                                                          2* If a symbol is to be drawn from
    """

    @abstractmethod
    def material_mapping(self, super_material):
        """
        Method given a set of Materials objects, creates the material_mapping dictionary, assigning labels
        to Materials as appropriate.

        Labels assigned to materials must correspond to materials labels appearing in symbol_mapping.
        A matching label means that the material holds the target Symbol.

        Args:
            materials (SuperMaterial): SuperMaterial containing candidate materials that can be plugged in to the model.
        Returns:
            (dict<str, Material>) OR None
                dictionary mapping String labels to Material objects or None if the mapping was unsuccessful.
        """
        pass
