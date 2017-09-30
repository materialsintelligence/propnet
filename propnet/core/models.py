from abc import ABCMeta, abstractmethod
from typing import List, Dict, Optional, Union, Tuple, Callable
from propnet.core.properties import PropertyName
from propnet.core.conditions import Condition
from propnet import ureg, QuantityLike
import sympy as sp

class AbstractModel(metaclass=ABCMeta):

    #__slots__ = ('equations', 'symbol_mapping', 'conditions',
    #             'assumptions', 'references', 'test_sets',
    #             'sympy_equations', 'evaluate')

    def __init__(self):

        # validate conditions

        # test symbol mapping, if symbol not in properties suggest adding it

        # wrap symbols with units

        # list sympy constraints as reserved words
        # we can add our own additional constraints to this list
        self._reserved_sympy_constraints = ('commutative',
                                            'complex',
                                            'imaginary',
                                            'real',
                                            'integer',
                                            'odd',
                                            'even',
                                            'prime',
                                            'composite',
                                            'zero',
                                            'nonzero',
                                            'algebraic',
                                            'trascendental',
                                            'irrational',
                                            'finite',
                                            'infinite',
                                            'negative',
                                            'nonnegative',
                                            'positive',
                                            'nonpositive',
                                            'hermitian',
                                            'antihermitian'
                                            )
        self.default_constraints = {
            'finite': True,
            'nonzero': True,
            'real': True
        }

        pass

    @abstractmethod
    def equations(self, **kwargs) -> List[Callable]:
        """
        A general method which, when provided with a collection of
        values and a desired output, will try to return that output.

        For analytical models, this should use sympy's solveset.

        :param values: symbols and their associated physical quantites
        :param output_symbol: symbol for desired output
        :return:
        """
        pass

    def _sympy_equations(self):
        """
        Converted user-provided equations into Sympy equations that
        can be solved. Uses user-provided symbol mapping.
        :return:
        """
        #symbols = sp.symbols(self.symbol_mapping.values())
        return NotImplementedError

    def _enforce_units(self, f: Callable) -> Callable:
        """
        Takes the solution from sympy, lambdaifies it,
        and wraps it with the appropriate units.

        :param f: Equation (Python or NumPy)
        :return: f wrapped with appropriate units
        """
        return NotImplementedError

    @property
    @abstractmethod
    def symbol_mapping(self) -> Dict[str, PropertyName]:
        """
        Maps symbols used to their canonical property names.
        :return:
        """
        pass

    @property
    @abstractmethod
    def valid_outputs(self) -> List[str]:
        """
        Model will not attempt to solve for a symbol that
        is not a valid output. This is useful for models
        that are known to be many-to-one functions that are
        not invertible.

        :return: a list of symbols that are valid outputs
        """
        pass

    @property
    @abstractmethod
    def conditions(self) -> Dict[str: List[Condition]]:
        """
        Required conditions that the model inputs should satisfy. If
        inputs are known not to satisfy a condition, the model will not
        run. If it is unknown if inputs satisfy a condition, then the
        condition will be added to the list of assumptions of the
        model output.

        :return: a mapping of symbols to a list of their conditions
        """
        return {}

    @property
    @abstractmethod
    def assumptions(self) -> List[Condition]:
        return []

    @property
    @abstractmethod
    def references(self) -> Optional[List[str]]:
        """
        :return: A list of references as BibTeX strings
        """
        pass

    @property
    @abstractmethod
    def test_sets(self) -> Dict[str, QuantityLike]:
        """
        Integrated test values. These are used by unit testing,
        and also when testing the model interactively

        :return:
        """
        return

    def evaluate(self,
                 values: Dict[str, QuantityLike],
                 desired_output: str):
        return NotImplementedError