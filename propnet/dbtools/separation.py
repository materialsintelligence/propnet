from maggma.builders import Builder
from maggma.utils import grouper
import pydash
from itertools import chain
from propnet import ureg
from propnet.core.registry import Registry
# noinspection PyUnresolvedReferences
import propnet.symbols


class SeparationBuilder(Builder):
    """
    Converts old-style propnet database into separate quantity-centered
    and materials-centered databases.
    """

    def __init__(self, propnet_store, quantity_store, material_store=None,
                 criteria=None, props=None, chunk_size=100):
        """

        Args:
            propnet_store (Mongolike Store): old-style propnet store
            quantity_store (Mongolike Store): store for quantities
            material_store (Mongolike Store): store for materials
            criteria (dict): JSON-style criteria for MongoDB find() query
            **kwargs: arguments to Builder parent class
        """
        self.material_store = material_store
        self.propnet_store = propnet_store
        self.quantity_store = quantity_store
        self.criteria = criteria
        self.total = None
        self.props = props or list(Registry("symbols").keys())

        super(SeparationBuilder, self).__init__(sources=[propnet_store],
                                                targets=[quantity_store, material_store],
                                                chunk_size=chunk_size)

    def get_items(self):
        # Borrowed from MapBuilder
        keys = self.propnet_store.distinct('task_id', criteria=self.criteria)
        containers = self.props + ['inputs']
        self.total = len(keys)
        for chunked_keys in grouper(keys, self.chunk_size, None):
            chunked_keys = list(filter(None.__ne__, chunked_keys))
            for doc in list(
                    self.propnet_store.query(
                        criteria={'task_id': {
                            "$in": chunked_keys
                        }},
                        properties=containers + ['task_id'],
                    )):
                yield doc

    def process_item(self, item):
        quantities = []
        material = item.copy()

        containers = [c + '.quantities' for c in self.props
                      if pydash.get(material, c)] + ['inputs']

        for container in containers:
            for q in pydash.get(material, container):
                this_q = q.copy()
                this_q['material_key'] = material['task_id']
                prov_inputs = pydash.get(this_q, 'provenance.inputs')
                if prov_inputs:
                    new_prov_inputs = [qq['internal_id'] for qq in prov_inputs]
                else:
                    new_prov_inputs = None
                pydash.set_(this_q, 'provenance.inputs', new_prov_inputs)
                quantities.append(this_q)

            pydash.set_(material, container,
                        [q['internal_id']
                         for q in pydash.get(material, container)])

            if container != 'inputs':
                prop = container.split(".")[0]
                units = Registry("units").get(prop)
                if units != pydash.get(material, [prop, 'units']):
                    pq_mean = ureg.Quantity(material[prop]['mean'],
                                            material[prop]['units']).to(units)
                    pq_std = ureg.Quantity(material[prop]['std'],
                                           material[prop]['units']).to(units)
                    q['mean'] = pq_mean.magnitude
                    q['std'] = pq_std.magnitude
                    q['units'] = pq_mean.units.format_babel()
            
        for q in quantities:
            units = Registry("units").get(q['symbol_type'])
            if q['units'] != units:
                pq = ureg.Quantity(q['value'], q['units']).to(units)
                q['value'] = pq.magnitude
                q['units'] = pq.units.format_babel()

        return quantities, material

    def update_targets(self, items):
        qs = [v[0] for v in items]
        qs = list(chain.from_iterable(qs))
        ms = [v[1] for v in items]

        self.quantity_store.update(qs, key='internal_id')
        self.material_store.update(ms, key='task_id')

    def finalize(self, cursor=None):
        q_indices = ['internal_id', 'symbol_type', 'data_type', 'material_key']
        m_indices = ['task_id']
        for idx in q_indices:
            self.quantity_store.ensure_index(idx)
        for idx in m_indices:
            self.material_store.ensure_index(idx)

        super().finalize(cursor)
