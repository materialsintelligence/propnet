from abc import ABCMeta, abstractmethod
from functools import wraps
from propnet.properties import PropertyType
from propnet.core.properties import Property
from propnet import logger
import sympy as sp

# typing information, for type hinting only
from typing import *
from propnet import ureg

# TODO: add pint integration
# TODO: decide on interface for conditions, assumptions etc.
# TODO: decide on interface for multiple-material models.


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
        names. Keys are symbols, values are canonical property names.

        Include mappings for all required inputs as well as conditions
        that must be checked for the model to function property.

        Because this is a dictionary, each symbol must be unique.
        """
        pass

    @property
    @abstractmethod
    def connections(self) -> Dict[str, Set[str]]:
        """
        Define your model inputs and outputs in terms of symbols
        listed in the symbol_mapping method.

        The model will not attempt to solve for an output
        unless it is marked as a valid output.

        Keys represent  potential output symbols,
        values represent a set of input symbols.

        :return: a dict of output symbols to a set of input symbols
        """
        pass

    @property
    @abstractmethod
    def constraint_properties(self) -> Set[str]:
        """
        Define additional model inputs, not included in connections,
        that must be examined for the model to ascertain whether it
        is valid in the given context.

        These inputs should be specified in terms of a set of symbols.

        :return: a set of symbols required for the model to check the
                 assumptions upon which it is based.
        """
        pass

    @abstractmethod
    def inputs_are_valid(self, input_props: Dict[str, Any]) -> bool:
        """
        Given a set of symbols and values (given by input_props) are fed
        into the model, returns whether the input values meet the assumptions
        required for the model to be valid.

        These input values include values of inputs specified in the connections
        method as well as values of inputs specified in the constraint_properties
        method.

        :return: a boolean value (T/F) signaling whether the combination of
                 model inputs and constraints are valid for the given model.
        """
        pass

    @abstractmethod
    def evaluate(self,
                 symbols_and_values_in: Dict[str, ureg.Quantity],
                 symbol_out: str) -> Optional[ureg.Quantity]:
        """
        Evaluate the model given a set of input values and a desired output.
        Evaluate will never be called unless the suite of inputs and assumptions
        passes the inputs_are_valid method.

        :param symbols_and_values_in: dict of symbols and their
               associated values used as inputs
        :param symbol_out: symbol for your desired output
        :return: value the model produces as output.
        """
        pass

    @abstractmethod
    def output_conditions(self, symbol_out: str) -> List[Any]:
        """
        Given the model is used to calculate the output symbol, symbol_out,
        returns a list of conditions that apply to the output object.

        :param symbol_out: symbol for your desired output
        :return: list of additional conditions that apply to the output
        """
        pass

    @property
    @abstractmethod
    def test_sets(self) -> List[Tuple[str, ureg.Quantity]]:
        """
        Add test sets to your model. These are used by unit testing,
        and also when testing the model interactively. A test set is
        a dict with keys that correspond to your symbols and values
        that are a (value, unit) tuple.
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

    @property
    def sp_vars(self) -> List[sp.Symbol]:
        """
        Helper method to convert all symbols of the AbstractAnalyticalModel instnace
        into SymPy variables.
        :return:
        """
        return list((sp.Symbol(x) for x in self.symbol_mapping.keys()))

    @property
    @abstractmethod
    def equations(self) -> List[str]:
        """
        List of equations constituting the abstract model using symbols indicated in
        the provided symbol_mapping method.
        Each equation must be formatted as an expression equal to zero. The strings
        returned must then reflect simply these expressions without an equals sign.
        :return: List of strings giving valid sympy expressions equal to zero
                 between the symbols.
        """
        pass

    @property
    def connections(self) -> Dict[str, Set[str]]:
        """
        Implements the abstract connections method for analytical models assuming that
        the equations are formulated such that every variable can be solved if all
        other variables are known.
        Please override this method in your particular subclass if this is not the case.

        :return: a dict of output symbols to a set of input symbols
        """
        symbols = list(self.symbol_mapping.keys())
        to_return = {}
        for i in range(0, len(symbols)):
            inputs = [None]*(len(symbols)-1)
            k = 0
            for j in range(0, len(symbols)):
                if j == i:
                    continue
                inputs[k] = symbols[j]
                k += 1
            to_return[i] = Set(inputs)
        return to_return

    def evaluate(self,
                 symbols_and_values_in: Dict[str, ureg.Quantity],
                 symbol_out: str) -> Optional[ureg.Quantity]:
        """
        Solve provided equations using SymPy.
        :param symbols_and_values_in: mapping of symbols to known values
        :param symbol_out: desired output
        :return: Sympy solution for the desired symbol_out
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