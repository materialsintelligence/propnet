from os import environ
from propnet.web.app import app
from uuid import uuid4

app.server.secret_key = environ.get('FLASK_SECRET_KEY', str(uuid4()))

if __name__ == '__main__':
    app.run_server(debug=True)