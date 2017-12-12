from propnet.core.models import AbstractAnalyticalModel

class HallPetchRelationship(AbstractAnalyticalModel):
    """ """

    @property
    def title(self):
        """ """
        return "Hall-Petch Relationship"

    @property
    def tags(self):
        """ """
        return ["mechancial"]

    @property
    def description(self):
        """

        Args:

        Returns:
          The Hall-Petch relationship relates yield stress with average grain size.

        """

    @property
    def references(self):
        """

        Args:

        Returns:
          @misc{petch2017,
          title={Hall-Petch relationship},
          url={https://en.wikipedia.org/wiki/Grain_boundary_strengthening#Hall.E2.80.93Petch_relationship},
          year={2017},
          month={Dec}}

        """

    @property
    def symbol_mapping(self):
        """ """
        return {
            'd': 'grain_diameter',
            'k': 'strengthening_coefficient',
            'o': 'dislocation_movement_stress',
            'y': 'yield_stress',
            's': 'structure'
        }

    @property
    def equations(self):
        """ """
        return ["y - o - k / d**(.5)"]
