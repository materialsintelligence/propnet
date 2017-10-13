from abc import ABCMeta, abstractmethod
from functools import wraps
from propnet.properties import PropertyType
from propnet import logger
import sympy as sp

# typing information, for type hinting only
from typing import *
from propnet import ureg

# TODO: add pint integration
# TODO: decide on interface for conditions, assumptions etc.

class AbstractModel(metaclass=ABCMeta):

    def __init__(self,
                 strict: bool = False):
        """
        Initialize a model, will retrieve/validate
        properties.

        :param strict: If strict, enforce input and output
        units (should be True for production).
        """

        # retrieve units for each symbol
        try:
            self.unit_mapping = {symbol:PropertyType[name].value.units
                                 for symbol, name in self.symbol_mapping.items()}
        except:
            raise ValueError("Please check your property names in your symbol mapping, "
                             "are they all valid?")

    @property
    @abstractmethod
    def title(self) -> str:
        """
        Add a title to your model.
        """
        pass

    @property
    @abstractmethod
    def tags(self) -> List[str]:
        """
        Add tags to your model as a list of strings.
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Add a description to your model as a multi-line string.
        """
        pass

    @property
    @abstractmethod
    def references(self) -> Optional[List[str]]:
        """
        :return: A list of references as BibTeX strings
        """
        pass

    @property
    @abstractmethod
    def symbol_mapping(self) -> Dict[str, str]:
        """
        Map the symbols used in your model to their canonical property
        names.
        """
        pass

    @property
    @abstractmethod
    def connections(self) -> Dict[str, Set[str]]:
        """
        Define your model inputs and outputs. The model will
        not attempt to solve for an output unless it is marked
        as a valid output.

        :return: a dict of output symbol to a set of input symbols
        """
        pass

    @property
    @abstractmethod
    def test_sets(self) -> List[Tuple[str, ureg.Quantity], List[Tuple[str, ureg.Quantity]]]:
        """
        Add test sets to your model. These are used by unit testing,
        and also when testing the model interactively. A test set is
        a dict with keys that correspond to your symbols and values
        that are a (value, unit) tuple.
        """
        pass

    @abstractmethod
    def evaluate(self,
                 symbols_and_values_in: Dict[str, ureg.Quantity],
                 symbol_out: str) -> Optional[ureg.Quantity]:
        """
        Evaluate the model.

        :param symbols_and_values_in: dict of symbols and their
        associated values used as inputs
        :param symbol_out: symbol for your desired output
        :return:
        """
        pass

    def test(self) -> bool:
        """
        Test the model using values in test_sets. Used by unit tests.

        :return: True if tests pass.
        """
        return NotImplementedError

        # TODO: add method to test a model interactively, that can also be used in unit tests

    #    for test_set in self.test_sets:

    #        for output in self.valid_outputs:

    #            if output in test_set.keys():

    #                correct_output = ureg.Quantity(test_set[output][0],
    #                                               test_set[output][1])

    #                symbols_and_values_in = test_set.copy()
    #                del symbols_and_values_in[output]
    #                for k, v in symbols_and_values_in.items():
    #                    symbols_and_values_in[k] = ureg.Quantity(v[0], v[1])

    #                calculated_output = self.evaluate(symbols_and_values_in, output)

    #                assert correct_output==calculated_output,\
    #                    "Test failed for test set: {}".format(test_set)
    #    return True



class AbstractAnalyticalModel(AbstractModel):
    """
    A Model for which we define all equations inside Propnet,
    and can solve them symbolically using SymPy.
    """

    @abstractmethod
    def equations(self, **kwargs) -> List[Callable]:
        """
        List of equations to solve with SymPy.
        :param kwargs: one kwarg for each symbol
        :return:
        """
        # defined for each analytical model
        pass

    def evaluate(self,
                 symbols_and_values_in: Dict[str, ureg.Quantity],
                 symbol_out: str) -> Optional[ureg.Quantity]:
        """
        Solve provided equations using SymPy.
        :param symbols_and_values_in: mapping of known values to
        their symbols
        :param symbol_out: desired output
        :return:
        """
        # common implementation for all analytical models
        return NotImplementedError

    def connections(self):
        # common implementation for all analytical models
        return NotImplementedError


def validate_evaluate(func):
    """
    A wrapper function to check that models conform to spec.
    :param func: an `evaluate` method
    :return:
    """
    @wraps(func)
    def validate_evaluate_wrapper(self, symbols_and_values_in, symbol_out):

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