# !/usr/bin/env python

from setuptools import setup

setup(
    name='propnet',
    packages=['propnet'],
    version='0.0',
    author='Propnet Development Team',
    author_email='matt@mkhorton.net',
    description='Materials Science models, pre-alpha.',
    url='https://github.com/materialsintelligence/propnet',
    download_url='https://github.com/materialsintelligence/propnet/archive/0.0.tar.gz',
    entry_points={"console_scripts": ["propnet = propnet.cli:main"]},
    install_requires=["dash-core-components>=0.22.1",
                      "dash-html-components>=0.10.1",
                      "dash-renderer>=0.12.1",
                      "dash-table-experiments>=0.6.0",
                      "frozendict==1.2",
                      "gunicorn>=19.7.1",
                      "habanero==0.6.0",
                      "monty==1.0.3",
                      "networkx>=2.0",
                      "numpy>=1.15.1",
                      "Pint==0.8.1",
                      "plotly==2.0.15",
                      "pybtex==0.21",
                      "pydash==4.5.0",
                      "pylatexenc==1.2",
                      "six>=1.11.0",
                      "pymatgen>=2018.5.22",
                      "pytest==3.2.3",
                      "Flask-Caching==1.3.3",
                      "uncertainties==3.0.2",
                      "maggma>=0.6.5",
                      "gbml>=1.1.0"]
)
