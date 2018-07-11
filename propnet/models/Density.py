from propnet.core.models import AbstractModel
from propnet import ureg


class Density(AbstractModel):
    """
    Returns the atomic density assuming that all sites in the sites dictionary of the structure
    correspond to a single atom in the crystal motif.
    """

    def evaluate(self, symbol_values):
        s = symbol_values['s']
        # TODO: please check this @dmrdjenovic
        return {'p': ureg.Quantity.from_tuple(
                    [len(s.sites)/s.volume, [['angstrom', -3]]]),
                'rho': float(s.density) * ureg.gram * ureg.centimeter ** -3,
                'successful': True}
