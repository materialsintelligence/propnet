from monty.json import MSONable
from propnet.core.quantity import BaseQuantity, QuantityFactory
from propnet.core.provenance import ProvenanceElement
from propnet import ureg
from propnet.symbols import DEFAULT_SYMBOLS, Symbol
import copy
import logging

logger = logging.getLogger(__name__)


class StorageQuantity(MSONable):
    """
    A class to hold quantity data intended for database storage.

    This class does not inherit from BaseQuantity in effort to remain autonomous as a storage
    container, and not implementing much of the functionality required by the BaseQuantity class.

    This class is the top-level object for storage. StorageQuantity objects contain a ProvenanceStore
    object for provenance which therein contains ProvenanceStoreQuantity objects as its inputs.

    Hierarchy for non-storage objects:
        BaseQuantity has a ProvenanceElement (self.provenance)
        ProvenanceElement has a list of BaseQuantity objects (self.provenance.inputs)
        Each input BaseQuantity object has a ProvenanceElement with BaseQuantity inputs, etc.

    Hierarchy for non-storage objects:
        StorageQuantity has a ProvenanceStore (self.provenance)
        ProvenanceStore has a list of ProvenanceStoreQuantity objects (self.provenance.inputs)
        Each input ProvenanceStoreQuantity object has a ProvenanceStore
            with ProvenanceStoreQuantity inputs, etc.

    The main purpose of these three classes is to prevent storing many copies of
    provenance quantity data (i.e. the values) in the database to save on document size.
    """

    def __init__(self, data_type=None, symbol_type=None,
                 value=None, units=None,
                 internal_id=None, tags=None, provenance=None,
                 uncertainty=None):
        """
        Constructor for StorageQuantity object.

        Note: In general, one should use the from_quantity() class method to generate these
        objects directly from BaseQuantity-derived objects.

        Args:
            data_type: (str) indicates what type of BaseQuantity object it was created from.
                Must be "NumQuantity" or "ObjQuantity".
            symbol_type: (Symbol) symbol representing the type of data contained in the object
            value: (id) the data stored in the object
            units: (pint.unit) unit object representing units of the value or None for non-
                numerical values
            internal_id: (str) unique identifier. (Note: this is used for lookup when the
                object is deserialized)
            tags: (list<str>) tags associated with the quantity, typically
                related to its origin, e. g. "DFT" or "ML" or "experiment"
            provenance: (ProvenanceElement or ProvenanceStore) provenance associated with the
                object. See BaseQuantity.__init__() for more info.
            uncertainty: (pint.Quantity or NumQuantity) uncertainty associated with the
                value stored in the same units
        """

        if isinstance(value, ureg.Quantity):
            value = copy.deepcopy(value)

        if isinstance(value, BaseQuantity):
            raise TypeError("Use from_quantity() for values of type BaseQuantity.")

        self._internal_id = internal_id
        self._data_type = data_type
        self._symbol_type = symbol_type
        self._tags = tags
        self._value = value
        self._units = units
        self.uncertainty = uncertainty
        self.provenance = provenance

    @property
    def uncertainty(self):
        """

        Returns:

        """
        # Returns copy so the class member remains immutable
        return copy.deepcopy(self._uncertainty)

    @uncertainty.setter
    def uncertainty(self, rhv):
        if isinstance(rhv, (BaseQuantity, StorageQuantity)):
            self._uncertainty = rhv.value.to(self.units)
        elif isinstance(rhv, ureg.Quantity):
            self._uncertainty = rhv.to(self.units)
        elif isinstance(rhv, (list, tuple)):
            self._uncertainty = ureg.Quantity.from_tuple(rhv)
        elif not rhv:
            self._uncertainty = None
        else:
            raise TypeError("Expected BaseQuantity, StorageQuantity, pint Quantity, or tuple. "
                            "Instead received {}".format(type(rhv)))

    @property
    def provenance(self):
        return self._provenance

    @provenance.setter
    def provenance(self, rhv):
        if isinstance(rhv, ProvenanceStore) or rhv is None:
            pass
        elif isinstance(rhv, ProvenanceElement):
            rhv = ProvenanceStore.from_provenance_element(rhv)
        else:
            raise TypeError("Expected ProvenanceElement or ProvenanceStore. "
                            "Instead received {}".format(type(rhv)))
        self._provenance = rhv

    # The following property methods are needed for __eq__()
    @property
    def symbol(self):
        return self._symbol_type

    @property
    def tags(self):
        return self._tags

    @property
    def value(self):
        # Return copy so class member remains immutable
        return copy.deepcopy(self._value)

    @property
    def magnitude(self):
        return self.value

    @property
    def units(self):
        return self._units

    @classmethod
    def from_quantity(cls, quantity_in):
        if isinstance(quantity_in, StorageQuantity):
            return copy.deepcopy(quantity_in)
        elif isinstance(quantity_in, BaseQuantity):
            data_type = type(quantity_in).__name__
        else:
            raise TypeError("Expected StorageQuantity or"
                            "object that inherits BaseQuantity, instead received {}"
                            .format(type(quantity_in)))

        return cls(data_type=data_type, symbol_type=quantity_in._symbol_type,
                   value=quantity_in.magnitude, units=quantity_in.units,
                   internal_id=quantity_in._internal_id, tags=quantity_in.tags,
                   provenance=quantity_in.provenance,
                   uncertainty=quantity_in.uncertainty)

    def to_quantity(self, lookup=None):
        if self.provenance is not None:
            provenance_in = self._provenance.to_provenance_element(lookup=lookup)
        else:
            provenance_in = None

        out = QuantityFactory.create_quantity(symbol_type=self._symbol_type, value=self._value, units=self._units,
                                              tags=self._tags,
                                              provenance=provenance_in,
                                              uncertainty=self._uncertainty)

        out._internal_id = self._internal_id
        return out

    def __hash__(self):
        return hash(self._internal_id)

    def __str__(self):
        return "<{}, {} {}, {}>".format(self.symbol.name, self.value, self.units, self.tags)

    def __repr__(self):
        return self.__str__()

    def __bool__(self):
        return bool(self.value)

    def as_dict(self):
        symbol = self._symbol_type
        if symbol.name in DEFAULT_SYMBOLS.keys() and symbol == DEFAULT_SYMBOLS[symbol.name]:
            symbol = self._symbol_type.name

        return {"@module": self.__class__.__module__,
                "@class": self.__class__.__name__,
                "internal_id": self._internal_id,
                "data_type": self._data_type,
                "symbol_type": symbol,
                "value": self._value,
                "units": self._units.format_babel() if self._units else None,
                "provenance": self._provenance,
                "tags": self.tags,
                "uncertainty": self._uncertainty.to_tuple() if self._uncertainty else None}

    @staticmethod
    def reconstruct_quantity(d, lookup):
        return StorageQuantity.from_dict(d).to_quantity(lookup)

    def __eq__(self, other):
        if isinstance(other, (StorageQuantity, BaseQuantity)):
            return self._internal_id == other._internal_id and \
                   self.provenance == other.provenance
        else:
            return NotImplemented


class ProvenanceStore(ProvenanceElement):
    """
    A class to hold provenance data for storage. It is held within a StorageQuantity
    or ProvenanceStoreQuantity object. The class provides methods to coerce ProvenanceElement
    objects for storage.

    The main purpose of these three classes is to prevent storing many copies of
    provenance quantity data (i.e. the values) in the database to save on document size.
    """
    def __init__(self, model=None, inputs=None, source=None):
        super(ProvenanceStore, self).__init__(model=model,
                                              source=source)

        self.inputs = inputs

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, rhv):
        if rhv is None:
            self._inputs = None
            return

        if isinstance(rhv, BaseQuantity):
            rhv = [rhv]

        if getattr(rhv, '__iter__') is None:
            raise TypeError("Expected iterable object or BaseQuantity."
                            "Instead received {}".format(type(rhv)))

        if not all(isinstance(q, (StorageQuantity, BaseQuantity)) for q in rhv):
            invalid_types = [type(q) for q in rhv if not isinstance(q, (StorageQuantity, BaseQuantity))]
            raise TypeError("Invalid object type(s) provided: {}".format(invalid_types))

        self._inputs = [ProvenanceStoreQuantity.from_quantity(q)
                        if isinstance(q, BaseQuantity) else q
                        for q in rhv]

    @classmethod
    def from_provenance_element(cls, provenance_in):
        if isinstance(provenance_in, ProvenanceStore):
            return copy.deepcopy(provenance_in)
        elif not isinstance(provenance_in, ProvenanceElement):
            raise TypeError("Expected input type ProvenanceElement. "
                            "Instead received {}".format(type(provenance_in)))

        return cls(model=provenance_in.model,
                   inputs=provenance_in.inputs,
                   source=provenance_in.source)

    def to_provenance_element(self, lookup=None):
        if self.inputs:
            inputs = [v.to_quantity(lookup=lookup) for v in self.inputs]
        else:
            inputs = None
        return ProvenanceElement(model=self.model,
                                 inputs=inputs,
                                 source=self.source)

    def __eq__(self, other):
        if type(other) is type(self):
            return self.model == other.model and \
                   set(self.inputs or []) == set(other.inputs or [])
        elif isinstance(other, ProvenanceElement) and issubclass(type(self), ProvenanceElement):
            return self.model == other.model and \
                   set(self.inputs or []) == set(self.from_provenance_element(other).inputs or [])
        else:
            return NotImplemented


class ProvenanceStoreQuantity(StorageQuantity):
    def __init__(self, data_type=None, symbol_type=None,
                 value=None, units=None,
                 internal_id=None, tags=None, provenance=None,
                 uncertainty=None, from_dict=False):

        super(ProvenanceStoreQuantity, self).__init__(data_type=data_type, symbol_type=symbol_type,
                                                      value=value, units=units,
                                                      internal_id=internal_id, tags=tags,
                                                      provenance=provenance,
                                                      uncertainty=uncertainty)
        self._from_dict = from_dict
        self._value_retrieved = value is not None

    def as_dict(self):
        symbol = self._symbol_type
        if symbol.name in DEFAULT_SYMBOLS.keys() and symbol == DEFAULT_SYMBOLS[symbol.name]:
            symbol = self._symbol_type.name

        return {"@module": self.__class__.__module__,
                "@class": self.__class__.__name__,
                "data_type": self._data_type,
                "symbol_type": symbol,
                "internal_id": self._internal_id,
                "tags": self._tags,
                "provenance": self._provenance
                }

    @classmethod
    def from_dict(cls, d):
        if isinstance(d['symbol_type'], dict):
            symbol = Symbol.from_dict(d['symbol_type'])
        elif isinstance(d['symbol_type'], str):
            symbol = d['symbol_type']
        else:
            raise TypeError("Unrecognized symbol data structure.")

        return cls(data_type=d['data_type'], symbol_type=symbol,
                   value=None,
                   units=None,
                   internal_id=d['internal_id'], tags=d['tags'],
                   provenance=ProvenanceStore.from_dict(d['provenance']),
                   uncertainty=None, from_dict=True)

    def is_from_dict(self):
        return self._from_dict

    def is_value_retrieved(self):
        return self._value_retrieved

    def lookup_value(self, lookup):
        lookup_fun = None
        if isinstance(lookup, dict):
            lookup_fun = lookup.get
        elif not callable(lookup) and lookup_fun is None:
            raise TypeError("Specified lookup is not callable or a dict.")
        else:
            lookup_fun = lookup

        d = lookup_fun(self._internal_id)

        if not d:
            logger.warning("Value not found for internal ID: {}".format(self._internal_id))
            return False

        if not isinstance(d, dict):
            raise TypeError("Expected dict, instead received: {}".format(type(d)))

        if not all(k in d.keys() for k in ('value', 'units', 'uncertainty')):
            raise ValueError("Callable does not return dict containing 'value', "
                             "'units', and 'uncertainty' keys")

        self._value = d['value']
        self._units = d['units']
        self.uncertainty = d['uncertainty']
        self._value_retrieved = True
        return True

    def to_quantity(self, lookup=None):
        copy_of_self = copy.deepcopy(self)
        if lookup:
            copy_of_self.lookup_value(lookup)

        if not copy_of_self.is_value_retrieved():
            if copy_of_self.is_from_dict():
                raise ValueError("No value has been looked up successfully for this quantity. "
                                 "Run lookup_value() first or make sure the specified lookup "
                                 "function or dict contains the internal ID of this quantity: {}"
                                 "".format(copy_of_self._internal_id))
            else:
                raise ValueError("Cannot create new BaseQuantity with no value. Property 'value' has no value, "
                                 "possibly because it was never looked up. Use lookup_value() or initialize an "
                                 "object with a value.")

        return super(ProvenanceStoreQuantity, copy_of_self).to_quantity(lookup=lookup)

    def __hash__(self):
        return super().__hash__()
