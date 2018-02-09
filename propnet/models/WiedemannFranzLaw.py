from propnet.core.models import AbstractModel

class WiedemannFranzLaw(AbstractModel):

    @property
    def constraints(self):
        """ """
        return {
            'is_metallic': lambda is_metallic: is_metallic is True,
        }