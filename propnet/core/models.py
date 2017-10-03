from abc import ABCMeta, abstractmethod
from typing import List, Dict, Optional, Union, Tuple, Callable
from propnet.core.properties import Property
from propnet import ureg, QuantityLike
import sympy as sp

class AbstractAnalyticalModel(metaclass=ABCMeta):

    __slots__ = ('title', 'description', 'tags', 'references',
                 'symbol_mapping', 'unit_mapping', 'equations',
                 'test_sets', 'evaluate', 'valid_outputs')

    def __init__(self):

        # retrieve units for each symbol
        try:
            self.unit_mapping = {symbol:Property(name.upper())
                                 for symbol, name in self.symbol_mapping.items()}
        except:
            raise ValueError("Please check your property names in your symbol mapping, "
                             "are they all valid?")

    @property
    def title(self) -> str:
        """
        Add a title to your model.
        """
        pass

    @property
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
    def symbol_mapping(self) -> Dict[str, Property]:
        """
        Map the symbols used in your model to their canonical property
        names.
        """
        pass

    @abstractmethod
    def equations(self, **kwargs) -> List[Callable]:
        """
        :param values: symbols and their associated physical quantities
        :param output_symbol: symbol for desired output
        :return:
        """
        pass

    @property
    def valid_outputs(self) -> List[str]:
        """
        Define your valid outputs. The model will not attempt
        to solve for a symbol that is not a valid output. This
        is useful for models that are known to be many-to-one
        functions that are not invertible.

        :return: a list of symbols that are valid outputs
        """
        return self.symbol_mapping.items()

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
    def test_sets(self) -> Dict[str, QuantityLike]:
        """
        Add test sets to your model. These are used by unit testing,
        and also when testing the model interactively. A test set is
        a dict with keys that correspond to your symbols and values
        that are a (value, unit) tuple.
        """
        pass

    def evaluate(self,
                 values: Dict[str, QuantityLike],
                 desired_output: str):
        return NotImplementedError