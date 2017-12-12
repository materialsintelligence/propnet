from setuptools import setup

exec (open('force_graph/version.py').read())

setup(
    name='force_graph',
    version=__version__,
    author='mkhorton',
    packages=['force_graph'],
    include_package_data=True,
    license='MIT',
    description='Interface to react-vis-force',
    install_requires=[]
)
