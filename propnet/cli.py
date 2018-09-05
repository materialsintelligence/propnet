import argparse
import logging
from uuid import uuid4
from propnet.core.models import EquationModel, PyModel
from propnet.models import DEFAULT_MODEL_DICT
from monty.serialization import loadfn
from propnet.web.app import app
from os import environ

__author__ = "Joseph Montoya <montoyjh@lbl.gov>"
__version__ = 0.1
__date__ = "Sep 4 2018"


log = logging.getLogger(__name__)
app.server.secret_key = environ.get('FLASK_SECRET_KEY', str(uuid4()))
server = app.server

def run_web_app(args):
    if args.debug:
        log.setLevel('DEBUG')
    app.run_server(debug=args.debug)


def validate(args):
    if args.name is not None:
        import nose; nose.tools.set_trace()
        model = DEFAULT_MODEL_DICT[args.name]
        if not args.test_data:
            model.validate_from_preset_test()
            print("Model validated with test data")
            return True
    elif args.file is not None:
        if args.file.endswith(".yaml"):
            EquationModel.from_file(args.file)
        elif args.file.endswith(".py"):
            # This should define config
            l, g = locals().copy(), globals().copy()
            with open(args.file) as f:
                code = compile(f.read(), args.file, 'exec')
                exec(code, globals())
            config = globals().get('config')
            model = PyModel(**config)

    if args.test_data is not None:
        td_data = loadfn(args.test_data)
        for td_datum in td_data:
            model.test(**td_datum)
        print("{} validated with test data".format(model.name))
    return True


def main():
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
    parser_validate.set_defaults(func=validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()