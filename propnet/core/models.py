import sympy as sp

# typing information, for type hinting only
from typing import *

from abc import ABCMeta, abstractmethod
from functools import wraps
from hashlib import sha256

from propnet.symbols import PropertyType
from propnet import logger
from propnet import ureg

# TODO: add pint integration
# TODO: decide on interface for conditions, assumptions etc.
# TODO: decide on interface for multiple-material models.

class AbstractModel(metaclass=ABCMeta):
    """ """

    def __init__(self,
                 strict: bool = False):
        """
        Initialize a model, will retrieve/validate
        properties.

        :param strict: If strict, enforce input and output
        units (should be True for production).
        """

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

        #if not len(self.test_sets):
        #    logger.warning('No test sets have been defined for model: {}'
        #                    .format(self.__class__.__name__))

        #if not self.outputs or not self.connections:
        #    raise ValueError('Model has to have connections (input/output symbols) specified.')

        #if len(self.title) > 80:
        #    logger.warning('Title is too long for model: {}'.format(self.__class__.__name__))

    @property
    @abstractmethod
    def title(self) -> str:
        """Add a title to your model."""
        pass

    @property
    @abstractmethod
    def tags(self) -> List[str]:
        """Add tags to your model as a list of strings."""
        pass

    @property
    def revision(self):
        """The current revision of the model.
        
        Override for subsequent revisions to model if model
        functionality changes.
        :return:

        Args:

        Returns:

        """
        return 1

    @property
    @abstractmethod
    def description(self) -> str:
        """Add a description to your model as a multi-line string."""
        pass

    @property
    @abstractmethod
    def references(self) -> Optional[List[str]]:
        """:return: A list of references as BibTeX strings"""
        pass

    @property
    @abstractmethod
    def symbol_mapping(self) -> Dict[str, str]:
        """Map the symbols used in your model to their canonical property
        names. Keys are symbols, values are canonical property names.
        
        Include mappings for all required inputs as well as conditions
        that must be checked for the model to function property.
        
        Because this is a dictionary, each symbol must be unique.

        Args:

        Returns:

        """
        pass

    @property
    def connections(self) -> Dict[str, Set[str]]:
        """Define your model inputs and outputs explicitly in terms
        of symbols listed in the symbol_mapping method.
        
        The model will not attempt to solve for an output
        unless it is marked as a valid output.
        
        Keys represent  potential output symbols,
        values represent a set of input symbols.
        
        :return: a dict of output symbols to a set of input symbols

        Args:

        Returns:

        """
        return None

    #@property
    #def constraint_properties(self) -> Set[str]:
    #    """
    #    Define additional model inputs, not included in connections,
    #    that must be examined for the model to ascertain whether it
    #    is valid in the given context.

    #    These inputs should be specified in terms of a set of symbols.

    #    :return: a set of symbols required for the model to check the
    #             assumptions upon which it is based.
    #    """
    #    return set()

    #def inputs_are_valid(self, input_props: Dict[str, Any]) -> bool:
    #    """
    #    Given a set of symbols and values (given by input_props) are fed
    #    into the model, returns whether the input values meet the assumptions
    #    required for the model to be valid.

    #    These input values include values of inputs specified in the connections
    #    method as well as values of inputs specified in the constraint_properties
    #    method.

    #    :return: a boolean value (T/F) signaling whether the combination of
    #             model inputs and constraints are valid for the given model.
    #    """
    #    return True

    @abstractmethod
    def evaluate(self,
                 symbols_and_values_in: Dict[str, ureg.Quantity],
                 symbol_out: str) -> Optional[ureg.Quantity]:
        """Evaluate the model given a set of input values and a desired output.
        Evaluate will never be called unless the suite of inputs and assumptions
        passes the inputs_are_valid method.

        Args:
          symbols_and_values_in: dict of symbols and their
        associated values used as inputs
          symbol_out: symbol for your desired output
          symbols_and_values_in: Dict[str: 
          ureg.Quantity]: 
          symbol_out: str: 

        Returns:
          value the model produces as output.

        """
        pass

    #def output_conditions(self, symbol_out: str) -> List[Any]:
    #    """
    #    Given the model is used to calculate the output symbol, symbol_out,
    #    returns a list of conditions that apply to the output object.
#
    #    :param symbol_out: symbol for your desired output
    #    :return: list of additional conditions that apply to the output
    #    """
    #    return []

    @property
    def test_sets(self) -> List[Tuple[str, Any]]:
        """Add test sets to your model. These are used by unit testing,
        and also when testing the model interactively. A test set is
        a dict with keys that correspond to your symbols and values
        that are a (value, unit) tuple.

        Args:

        Returns:

        """
        return []

    @property
    def _test_sets(self) -> List[Tuple[str, ureg.Quantity]]:
        """Parses test_sets into Quantities if they've been supplied
        as strings or tuples, e.g. "1 GPa" or (1, "GPa")
        :return:

        Args:

        Returns:

        """
        return NotImplementedError

    def __hash__(self):
        """
        A unique model identifier, function of model class name.
        :return (str):
        """
        return sha256(self.__class__.__name__.encode('utf-8')).hexdigest()[0:4]

    @property
    def model_id(self):
        """ """
        return "{}{}".format(self.__hash__(),
                             ' rev-{}'.format(self.revision) if self.revision > 1 else '')

    def test(self) -> bool:
        """Test the model using values in test_sets. Used by unit tests.
        
        :return: True if tests pass.

        Args:

        Returns:

        """
        return NotImplementedError


class AbstractAnalyticalModel(AbstractModel):
    """A Model for which we define all equations inside Propnet,
    and can solve them symbolically using SymPy.

    Args:

    Returns:

    """

    @property
    def sp_vars(self) -> List[sp.Symbol]:
        """Helper method to convert all symbols of the AbstractAnalyticalModel instnace
        into SymPy variables.
        :return:

        Args:

        Returns:

        """
        return list((sp.Symbol(x) for x in self.symbol_mapping.keys()))

    @property
    @abstractmethod
    def equations(self) -> List[str]:
        """List of equations constituting the abstract model using symbols indicated in
        the provided symbol_mapping method.
        Each equation must be formatted as an expression equal to zero. The strings

        Args:

        Returns:
          :return: List of strings giving valid sympy expressions equal to zero
          between the symbols.

        """
        pass

    #@property
    #def connections(self) -> Dict[str, Set[str]]:
    #    """
    #    Implements the abstract connections method for analytical models assuming that
    #    the equations are formulated such that every variable can be solved if all
    #    other variables are known.
    #    Please override this method in your particular subclass if this is not the case.

    #    :return: a dict of output symbols to a set of input symbols
    #    """
    #    symbols = list(self.symbol_mapping.keys())
    #    to_return = {}
    #    for i in range(0, len(symbols)):
    #        inputs = [None]*(len(symbols)-1)
    #        k = 0
    #        for j in range(0, len(symbols)):
    #            if j == i:
    #                continue
    #            inputs[k] = symbols[j]
    #            k += 1
    #        to_return[i] = set(inputs)
    #    return to_return

    def evaluate(self,
                 symbols_and_values_in: Dict[str, ureg.Quantity],
                 symbol_out: str) -> Optional[ureg.Quantity]:
        """Solve provided equations using SymPy.

        Args:
          symbols_and_values_in: mapping of symbols to known values
          symbol_out: desired output
          symbols_and_values_in: Dict[str: 
          ureg.Quantity]: 
          symbol_out: str: 

        Returns:
          Sympy solution for the desired symbol_out

        """
        ancillary_eqs = list((f'k-({v})' for (k, v) in symbols_and_values_in))        ###TODO extract float from pint? Discussion req. about pint integration here.
        vals_out = sp.solve(self.equations + ancillary_eqs)
        if not isinstance(vals_out, list):
            vals_out = [vals_out]
        to_return = []
        for entry in vals_out:
            if isinstance(entry, dict):
                exists = entry.get(symbol_out)
                if exists is not None:
                    to_return.append(entry.get(symbol_out))
        for i in range(0, len(to_return)):
            if 'evalf' in dir(to_return[i]):
                to_return[i] = to_return[i].evalf()
        return to_return


def validate_evaluate(func):
    """A wrapper function to check that models conform to spec.

    Args:
      func: an `evaluate` method

    Returns:

    """
    @wraps(func)
    def validate_evaluate_wrapper(self, symbols_and_values_in, symbol_out):
        """

        Args:
          symbols_and_values_in: 
          symbol_out: 

        Returns:

        """

        # check we support this combination of inputs/outputs
        inputs = set(symbols_and_values_in.keys())
        if symbol_out not in self.connections:
            logger.error("The {} model does not support this output ({})."
                         "".format(self.name, symbol_out))
        else:
            if not self.connections[symbol_out].issubset(inputs):
                logger.error("The {} model does not support the output {} "
                             "for this combination of inputs ({}).".format(self.name,
                                                                           symbol_out,
                                                                           inputs))

        # check our units
        # TODO: make this more robust, check outputs too
        for symbol, value in symbols_and_values_in.items():
            if not isinstance(value, ureg.Quantity):
                logger.warn("Units are not defined for your {}, "
                            "using assumed units from property definition.".format(symbol))
                unit = getattr(self, 'unit_mapping')[symbol]
                symbols_and_values_in[symbol] = ureg.Quantity(value, unit)

        return func(self, symbols_and_values_in, symbol_out)

    return validate_evaluate_wrapper