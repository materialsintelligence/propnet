from abc import ABCMeta, abstractmethod
from typing import List, Dict, Optional, Union, Tuple
from propnet.core.properties import PropertyName
from propnet.core.conditions import Condition
from propnet import ureg

class AbstractModel(metaclass=ABCMeta):

    # __slots__ = ...

    def __init__(self):

        # validate conditions

        # test symbol mapping, if symbol not in properties suggest adding it

        # wrap symbols with units

        self.symbols = self.symbol_mapping.values()

        pass


    # TODO: wrap this so that output_symbol is checked against valid outputs
    # TODO: wrap this so that input str are automatically converted to quantities
    # or just add additional method to define using strings?/np arrays?
    # this is probably in the pint documentation ...
    @abstractmethod
    def master_equations(self,
                         values: Dict[str, Union[ureg.Quantity, str, Tuple]],
                         output_symbol) -> ureg.Quantity:
        """
        A general method which, when provided with a collection of
        values and a desired output, will try to return that output.

        For analytical models, this should use sympy's solveset.

        :param values: symbols and their associated physical quantites
        :param output_symbol: symbol for desired output
        :return:
        """
        pass

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
    # TODO: add wrapped to convert str to ureg.Quantity
    def test_sets(self) -> Dict[str, Union[str, ureg.Quantity, Tuple]]:
        """
        Integrated test values. These are used by unit testing,
        and also when testing the model interactively

        :return:
        """
        return

    # TODO: will need an internal function here to wrap model outputs as PropertyInstances
    # (this will be clearer once we have the graph set up)
    #def _get_output(self, property_instances) -> List[PropertyInstances]:
    #    pass