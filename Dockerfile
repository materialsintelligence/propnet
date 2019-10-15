FROM python:3.7-buster
LABEL maintainer="mkhorton@lbl.gov"

RUN mkdir -p /home/project/dash_app
WORKDIR /home/project/dash_app

RUN apt-get update
RUN apt-get install graphviz libgraphviz-dev -y
RUN pip install numpy scipy pygraphviz
ADD requirements.txt /home/project/dash_app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


ENV PROPNET_NUM_WORKERS=8

# this can be obtained from materialsproject.org
ENV PMG_MAPI_KEY='MATERIALS_PROJECT_KEY_HERE'
ENV PROPNET_CORRELATION_STORE_FILE="CORRELATION_STORE_FILE_OR_JSON_STRING"

ADD . /home/project/dash_app

EXPOSE 8000
CMD gunicorn --workers=$PROPNET_NUM_WORKERS --timeout=300 --bind=0.0.0.0 app:server