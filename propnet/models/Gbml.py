from propnet.core.models import AbstractModel


class Gbml(AbstractModel):

    @property
    def title(self):
        return "Machine-learned bulk and shear modulus estimates"

    @property
    def tags(self):
        return ["machine learning"]

    @property
    def description(self):
        return """This model uses Gradient Boosting Machine-Locfit (GBML) to give
        predictions for bulk and shear moduli given material descriptors and training data.
        """

    @property
    def references(self):
        return """
        
        @article{deJong2016,
  doi = {10.1038/srep34256},
  url = {https://doi.org/10.1038/srep34256},
  year  = {2016},
  month = {Oct},
  publisher = {Springer Nature},
  volume = {6},
  number = {1},
  author = {Maarten de Jong and Wei Chen and Randy Notestine and Kristin Persson and Gerbrand Ceder and Anubhav Jain and Mark Asta and Anthony Gamst},
  title = {A Statistical Learning Framework for Materials Science: Application to Elastic Moduli of k-nary Inorganic Polycrystalline Compounds},
  journal = {Scientific Reports}
}
        
        """

    @property
    def symbol_mapping(self):
        return {
            'K': 'bulk_modulus',
            'G': 'shear_modulus',
            'formula': 'pretty_formula',
            'nsites': 'nsites',
            'volume': 'volume_unit_cell',
            'energy_per_atom': 'energy_per_atom'
        }


    @property
    def connections(self):
        return {
            frozenset({'K', 'G'}): {'formula', 'nsites', 'volume', 'energy_per_atom'}
        }

    def evaluate(self,
                 symbols_and_values_in,
                 symbol_out):

        # waiting for gbml package to support Python 3
        return NotImplementedError