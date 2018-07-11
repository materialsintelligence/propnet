from os import environ
from propnet.web.app import app
from uuid import uuid4
import argparse
import logging

log = logging.getLogger(__name__)

app.server.secret_key = environ.get('FLASK_SECRET_KEY', str(uuid4()))
server = app.server

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", action="store_true",
                           help="debug mode")
    args = argparser.parse_args()
    if args.debug:
        log.setLevel('DEBUG')
    app.run_server(debug=args.debug)