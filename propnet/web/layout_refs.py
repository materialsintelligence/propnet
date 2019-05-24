import dash_core_components as dcc
import dash_html_components as html


def refs_layout(app):
    layout = dcc.Markdown("""
## Funding

The _propnet_ project is part of the Accelerated Materials Design and Discovery program, funded by
[Toyota Research Institute](https://www.tri.global/research/).

## Sources
The propnet codebase uses data and functionality from a number of open-source databases and repositories.
For more detailed information regarding these sources, please see the following references. For reference
information on individual properties and models, please see the respective model or symbol detail page.

##### The Materials Project
> Anubhav Jain, Shyue Ping Ong, Geoffroy Hautier, Wei Chen, William Davidson Richards, Stephen Dacek, Shreyas Cholia, Dan Gunter, David Skinner, Gerbrand Ceder, and Kristin A\. Persson\.
The Materials Project: A materials genome approach to accelerating materials innovation\.
*APL Materials*, 1\(1\):011002, 2013\.
URL: [https://aip\.scitation\.org/doi/10\.1063/1\.4812323](https://aip.scitation.org/doi/10.1063/1.4812323), [doi:10\.1063/1\.4812323](https://doi.org/10.1063/1.4812323)\.  

##### Pymatgen
> Shyue Ping Ong, William Davidson Richards, Anubhav Jain, Geoffroy Hautier, Michael Kocher, Shreyas Cholia, Dan Gunter, Vincent L\. Chevrier, Kristin A\. Persson, and Gerbrand Ceder\.
Python materials genomics \(pymatgen\): a robust, open\-source python library for materials analysis\.
*Computational Materials Science*, 68:314–319, Feb 2013\.
URL: [https://doi\.org/10\.1016%2Fj\.commatsci\.2012\.10\.028](https://doi.org/10.1016%2Fj.commatsci.2012.10.028), [doi:10\.1016/j\.commatsci\.2012\.10\.028](https://doi.org/10.1016/j.commatsci.2012.10.028)\.  

##### AFLOW / AFLUX
> Stefano Curtarolo, Wahyu Setyawan, Gus L\.W\. Hart, Michal Jahnatek, Roman V\. Chepulskii, Richard H\. Taylor, Shidong Wang, Junkai Xue, Kesong Yang, Ohad Levy, Michael J\. Mehl, Harold T\. Stokes, Denis O\. Demchenko, and Dane Morgan\.
AFLOW: an automatic framework for high\-throughput materials discovery\.
*Computational Materials Science*, 58:218–226, Jun 2012\.
URL: [https://doi\.org/10\.1016%2Fj\.commatsci\.2012\.02\.005](https://doi.org/10.1016%2Fj.commatsci.2012.02.005), [doi:10\.1016/j\.commatsci\.2012\.02\.005](https://doi.org/10.1016/j.commatsci.2012.02.005)\.  

> Richard H\. Taylor, Frisco Rose, Cormac Toher, Ohad Levy, Kesong Yang, Marco Buongiorno Nardelli, and Stefano Curtarolo\.
A RESTful API for exchanging materials data in the AFLOWLIB\.org consortium\.
*Computational Materials Science*, 93:178–192, Oct 2014\.
URL: [https://doi\.org/10\.1016%2Fj\.commatsci\.2014\.05\.014](https://doi.org/10.1016%2Fj.commatsci.2014.05.014), [doi:10\.1016/j\.commatsci\.2014\.05\.014](https://doi.org/10.1016/j.commatsci.2014.05.014)\.  

> Frisco Rose, Cormac Toher, Eric Gossett, Corey Oses, Marco Buongiorno Nardelli, Marco Fornari, and Stefano Curtarolo\.
AFLUX: the LUX materials search API for the AFLOW data repositories\.
*Computational Materials Science*, 137:362–370, Sep 2017\.
URL: [https://doi\.org/10\.1016%2Fj\.commatsci\.2017\.04\.036](https://doi.org/10.1016%2Fj.commatsci.2017.04.036), [doi:10\.1016/j\.commatsci\.2017\.04\.036](https://doi.org/10.1016/j.commatsci.2017.04.036)\.  

##### AFLUX API Python Wrapper
> Conrad W\. Rosenbrock\.
A practical Python API for querying AFLOWLIB\.
2017\.
[arXiv:arXiv:1710\.00813](https://arxiv.org/abs/arXiv:1710.00813)\.""")

    return layout