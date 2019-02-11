"""
A simple command-line interface to propnet
"""

import argparse
import logging
import pdb
from uuid import uuid4
from os import environ
from propnet.core.models import EquationModel, PyModel
from monty.serialization import loadfn

import propnet.models
from propnet.core.registry import Registry

__author__ = "Joseph Montoya <montoyjh@lbl.gov>"
__version__ = 0.1
__date__ = "Sep 4 2018"

LOG = logging.getLogger(__name__)


def run_web_app(args):
    """Runs web app"""
    from propnet.web.app import app
    app.server.secret_key = environ.get('FLASK_SECRET_KEY', str(uuid4()))
    if args.debug:
        LOG.setLevel('DEBUG')
    app.run_server(debug=args.debug)


def validate(args):
    """Validates test data"""
    if args.name is not None:
        model = Registry("models")[args.name]
        if not args.test_data:
            test_wrapper(model.validate_from_preset_test, pdb)
            print("Model validated with test data")
            return True
    elif args.file is not None:
        if args.file.endswith(".yaml"):
            EquationModel.from_file(args.file)
        elif args.file.endswith(".py"):
            # This should define config
            with open(args.file) as this_file:
                code = compile(this_file.read(), args.file, 'exec')
                exec(code, globals())
            config = globals().get('config')
            model = PyModel(**config)

    if args.test_data is not None:
        td_data = loadfn(args.test_data)
        for td_datum in td_data:
            test_wrapper(model.test, args.pdb, **td_datum)
        print("{} validated with test data".format(model.name))
    return True


def test_wrapper(func, pdb, *args, **kwargs):
    """Simple wrapper that allows for pdb if chosen"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if pdb:
            pdb.post_mortem()
        raise(e)


def main():
    """Main body for CLI"""
    parser = argparse.ArgumentParser(description="""
    propnet is a command-line interface to the propnet python package""",
                                     epilog="""
    Author: Joseph Montoya
    Version: {}
    Last updated: {}""".format(__version__, __date__))
    subparsers = parser.add_subparsers()

    # Web runner
    parser_web = subparsers.add_parser("run_web", help="Run web server")
    parser_web.add_argument("-d", "--debug", help="Run in debug mode",
                            action="store_true")
    parser_web.set_defaults(func=run_web_app)

    # Validate models
    parser_validate = subparsers.add_parser(
        "validate", help="validate models")
    parser_validate.add_argument("-n", "--name", help="Default model name")
    parser_validate.add_argument("-f", "--file", help="Non-default model filename")
    parser_validate.add_argument("-t", "--test_data", help="Test data file")
    parser_validate.add_argument("--pdb", help="Invoke debugger on failure",
                                 action="store_true")
    parser_validate.set_defaults(func=validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
