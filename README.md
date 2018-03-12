# Propnet

Propnet is a knowledge graph for materials science. Given a set of *properties* about a given material, it can try to derive additional properties of that material using *models*. It is intended to integrated closely with the [Materials Project](http://materialsproject.org) and its large database of materials properties.

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/0803e3ca599a4250bb3ec2bb16b4b911)](https://www.codacy.com/app/mkhorton/propnet?utm_source=github.com&utm_medium=referral&utm_content=materialsintelligence/propnet&utm_campaign=badger)
[![Build Status](https://travis-ci.org/materialsintelligence/propnet.svg?branch=master)](https://travis-ci.org/materialsintelligence/propnet) [![Heroku](https://heroku-badge.herokuapp.com/?app=propnet&svg=1)](https://propnet.herokuapp.com) [![Docs](https://readthedocs.org/projects/propnet/badge/?version=latest)](https://readthedocs.org/projects/propnet/badge/?version=latest)

**This is not ready for public use yet, please wait for a formal release announcement/publication! Thank you.** Any questions should be directed to the project lead @computron or, in the case of code, to the individual project contributors.

For a design overview, please see the `docs/design_overview.md` file -- this is likely to change.

# Table of Contents

* [Installation and Use](#installation-and-use)
* [Guidelines for Contributions](#guidelines-for-contributions)
  * [Submitting a Property](#submitting-a-property)
  * [Submitting a Model](#submitting-a-model)
  * [Submitting a code contribution](#submitting-a-code-contribution)
* [Web Interface](#web-interface)
* [Contributors](#contributors) 
* [Acknowledgements](#acknowledgements)

# Installation and Use

Currently, a lot of ideas are still in flux, and there are no guarantees!

The recommended way to install and use the package is to clone the package locally and run `pipenv install`, followed by `pipenv shell` to launch Propnet inside a virtual environment.

To run Propnet inside a Jupyter notebook, run:

```bash
pipenv install ipykernel
pipenv shell
python -m ipykernel install --user --name=propnet-dev
jupyter notebook
```

A `propnet-dev` kernel will then be available inside Jupyter.


# Guidelines for Contributions

Properties and models each have their own distinct file, to make it easier to version models and properly credit the people who made them. When necessary, we will use the git short hash to refer to a specific version of a model or property, though in most cases these should not change much once they are created.

## Submitting a Symbol

All symbols can be found in `/symbols/`. These are split into `properties`, for material properties (these are typically scalars, vectors or tensors, and have physical units attached), `conditions` (essentially free variables, such as temperature, or applied stress) and `objects` (for example, crystallographic structure or formula).

Please copy an existing symbol/property and submit a pull request, its filename should match the canonical name of the property. Properties are defined in [YAML](http://yaml.org) syntax.

Key fields are as follows:

* `name`: A unique, canonical name for the property, lowercase, must be a valid Python identifier (no spaces)
* `unit`: A list of of lists, from [pint's serialization format](http://pint.readthedocs.io/en/latest/serialization.html)
* `display_names`: List of human-readable name(s), LaTeX syntax allowed, first name will be the preferred name
* `display_symbols`: As above, but for symbols
* `dimension`: The expected dimension (using the same definition as [numpy ndarray shape](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.shape.html)) of the property. If there are multiple ways to define the property (e.g. vector, scalar), define multiple properties with a simple model to convert between them.
* `test_value`: A physically-plausible test value in the units specified above, will be used to populate the graph with test data.
* `comment`: Optional string with additional information.

If you want to check your unit definitions, the easiest way is to try it interactively:

```
from pint import UnitRegistry
ureg = UnitRegistry()
# replace "..." with your units here
my_units = ureg.parse_expression("...")
# check units look correct
print(my_units)
# and convert to tuple
my_units = my_units.to_tuple()
```

## Submitting a Model

All models can be found in `/models/`.

Please copy an existing model and submit a pull request. There are typically two files associated with a model: a .py file containing the class (a subclass of `AbstractModel`) and any relevant logic (if applicable), and a .yaml file containing metadata (title, description, inputs/outputs, and the like) that is automatically loaded into the class provided its filename matches the model class name. If a model is a simple analytical equation, an equation can be specified in the .yaml file and this will be solved automatically.

The model metadata is as follows:

* `title`, `tags`, `description` and `references` for documenting the model, `tags` is a list of strings of whatever seems like a sensible tag for the model (for example 'mechanical', 'thermo' or 'electronic'), 'stub' is a special tag for trivial models that simply convert one representation of a property to a different equivalent representation, for example due to different notations etc., `references` should be a list of strings in BibTeX format
* `symbol_mapping`: for convenience, your model should use short `symbols` to represent quantities, e.g. `E` for `youngs_modulus`, the symbol mapping gives the canonical property name for each symbol
* `connections` gives the valid inputs and outputs for the model
* `evaluate` takes a dictionary of symbols and their associated values for inputs, and a desired `output_symbol`, and will return the value for the output symbol if possible, or `None` if it cannot be solved for that output symbol

In addition to analytical equations, as the project progresses we expect to support the following machine learning models:

* Neural networks (feedforward, convolutional, recurrent) via Caffe or Keras
* Scikit-learn models (tree ensembles, support vector machines, generalized linear models, feature engineering, pipeline models)

More documentation is pending!

## Submitting a code contribution

We only have a few guidelines at present:

* Please be [PEP8](https://www.python.org/dev/peps/pep-0008/) compliant (not strict about line length, but try to keep docstrings to 72 characters, code to 100)
* We're targeting Python 3.6+ only
* We're hoping to add [type hints](https://www.python.org/dev/peps/pep-0484/) for the core code
* If a function or method returns multiple values, return a `namedtuple`, this allows easier refactoring later and more readable code. Functions and methods should also always consistently return the same types (with the exception of `None`).
* If you spot a bad practice / anti-pattern in the code, you're honor bound to report it :-)

# Web Interface

Each model and property will have an auto-generated page associated with them. This page will allow manual input of model parameters and calculation of model outputs, and will also provide documentation/explanations of how the model works. The uses the [Dash library by plot.ly](https://plot.ly/products/dash/). To test it out, you can run the `app.py`, or can see the app deployed online at [https://propnet.herokuapp.com](https://propnet.herokuapp.com).

# Contributors

@computron, @mkhorton, @vtshitoyan, @LeighWeston86, @jdagdelen

# Acknowledgements

Hat tip to @jdagdelen for the Propnet name.
