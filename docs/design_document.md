# Table of Contents

* [Propnet Design Overview](#propnet-design-overview)
    * [Glossary](#glossary)
    * [The Graph](#the-graph)
    * [Property](#property)
    * [Material](#material)
    * [TODO: Constraints and Assumptions](#constraints-and-assumptions)
    * [Solving the Graph](#solving-the-graph)

# Propnet Design Overview

Propnet is designed to synthesize all current materials science knowledge into a graph, that can be traversed to calculate additional properties when provided with some initial information.

## Glossary

**Material:** Can be a formula, crystallographic structure, or more complex material

**PropertyType:** A distinct physical *Quantity,* such as lattice parameter or temperature

**Property:** A class containing a *Quantity* and its associated *PropertyType*, as well as any associated metadata such as references, that is typically associated with a *Material* (this may be renamed later)

**Quantity:** Refers to a *value* which can be a scalar/vector/matrix/tensor and its associated *unit*

**Model:** Something that relates one set of properties to another set of properties, an **AnalyticalModel** refers to a model governed by a set of equations that we try to solve internally using Propnet, otherwise a model can use external Python code, a ML model, etc. to calculate a given property

**Node:** All properties, materials, models are nodes in our graph.

**Edge:** An edge refers to a relationship between a given property, material or model and another property, material or model. An edge can have a direction and additional associated information.

## The Graph

As much as possible, information is encoded into the graph itself, rather than creating additional data structures. Rather than re-invent the wheel, we are using the popular and well-tested [networkx](https://networkx.github.io) graph library to provide the necessary data structures and graph algorithms that form the foundation of Propnet.

Our graph has the following types of node:

![Node types](docs/images/readme_node_types.svg)

## Property

A 'property' refers to a property of a material, such as its lattice parameter, band gap or bulk modulus.

In Propnet, properties are rigorously defined to ensure consistent interpretations between different models, and this crucially includes explicit units and uncertainties. Where uncertainties are unknown, we apply rule-of-thumb uncertainties derived from statistical analysis of similar data to give a first-approximation of our confidence in a property.

In terms of implementation, a `PropertyType` provides information on the property itself (e.g. lattice parameter has units of length), and a `Property` is an object that combines a `PropertyType` with a value (e.g. 5Å), a reference for where the property came from, and edges to the materials it is associated with.

![Simple property](docs/images/readme_property_simple.svg)

Once a material has been specified, we can define a property of that material, such as its lattice parameter. Above, the `Property` node contains the value of that property, and it points to the graph's canonical `PropertyType` node that the value describes.

![Dependent property](docs/images/readme_property_dependent.svg)

A `Property` with an edge to another `Property ` describes a dependent property: in this case, it could be the temperature at which that measurement was taken. For the first version of Propnet, all measurements are derived from 0 K DFT simulations, so this can be assumed implicitly.

![Multiple properties](docs/images/readme_property_multiple.svg)

It is also possible to define multiple properties for the same material, for example an experimental and DFT value. When evaluating the graph, the user can choose a strategy which will select the most appropriate property: for example, the user might select to always prefer DFT data, or to always prefer the property with the smallest uncertainty.

## Material

There is a necessary distinction between a real 'material' vs. an ideal, perfect crystal. In the [Materials Project](http://materialsproject.org/)  a 'material' typically refers to the latter. However, properties can come from many sources, including experiment data where additional phases or impurities might be present.

Propnet attempts to gracefully handled this distinction. A `Property` can have edges to the material it's associated with. In the simplest case, this could be an edge to a material defined simply by its chemical formula. If the structure is known, it is represented by a [pymatgen](https://github.com/materialsproject/pymatgen) `Structure` object.

![Simple materials](docs/images/readme_materials_simple.svg)

This shows the simplest definition of a material, either known by its chemical formula, or by its crystallographic structure.

![Compound materials](docs/images/readme_materials_compound.svg)

We can also represent compound materials. On the left, the edge attribute *x* gives the proportion of each structure that makes up the material. On the right, we use an edge attribute to define an orientation relationship between structures.

![Complex materials](docs/images/readme_materials_complex.svg)

In principle, we can construct materials of arbitrary complexity: here we represent a multi-layer material with film thicknessness specified by the edge attribute *t*, with one component of the multi-layer being doped.

At this stage, none of the materials in Propnet will be this complex, and a way of defining, for example, orientation relationships has not yet been decided on. This example is simply included as an illustration that the design should be forward-looking and transferrable.

## Constraints and Assumptions

**TODO: the ideas here will likely change, and are not implemented yet.**

An `Assumption` is associated with a property and can be something like:

* An `IsotropicAssumption`, assumes a material is isotropic
* A `Temperature(300)` assumption, assumes an experimental measurement was taken at room temperature

A `Constraint` is a model requirement and can be something like:

* A model could expect a lattice parameter `Property` with the condition that the property has an edge to an additional property that defines the temperature the lattice parameter was measured at.

If a model has inputs where it is unknown if they satisfy its constraints (e.g. a lattice parameter property exists in the graph, but the temperature it was measured at is not specified), then it becomes an `Assumption` in the model output.

Any reference or `Assumption` associated with a `Property` is propagated through the model and through the graph, so that when the graph is queried for the specific value of a property, the assumptions and a list of references are also supplied, in addition to the value of the property itself.

## Solving the Graph

![Model node](docs/images/readme_model_simple.svg)

The simplest `Model` relates one `PropertyType` to another `PropertyType`, as shown above.

![Full graph](docs/images/readme_full_graph.svg)

Incorporating all the models and node types into the full graph, we can see a clear separation between data (`Materials` and their associated `Properties`), and logic (`Models` and their associated `PropertyTypes`, which form the input and output of `Models`).

In practice, the 'logic layer' is the same for all Propnet instances, while the 'data layer' usually comes from a `Material` graph, which can be composed into the Propnet graph.

To 'solve' the graph, we look for `PropertyType` nodes that do not have any associated `Properties`, and then from the graph topology see if there is a route to calculate this property. If a route exists, the property is calculated, and a new `Property` inserted into the graph. The direction of the edge indicates whether the property is an input or output from a given model.