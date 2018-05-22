from propnet.core.models import AbstractModel

class PymatgenStructureProperties(AbstractModel):

    def plug_in(self, symbol_values):

        structure = symbol_values['structure']

        output = {}

        for attr in self.symbol_mapping.keys():
            if hasattr(structure, attr):
                output[attr] = getattr(structure, attr, None)

        return output