import dash_core_components as dcc
import dash_html_components as html


def home_layout():

    return dcc.Markdown("""

**Not intended for public use at this time.**
Internal collaborators only. Please report bugs :)

Our Github repository is available [here](https://github.com/materialsintelligence/propnet),
if you would like to test propnet locally please clone the repository, `pip install -r requirements.txt`
and `python setup.py develop`. 

Funded by: Accelerated Materials Design and Discovery, Toyota Research Institute
    
""")
