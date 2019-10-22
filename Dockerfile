FROM python:3.7-buster
LABEL maintainer="mkhorton@lbl.gov"

RUN mkdir -p /home/project/dash_app
WORKDIR /home/project/dash_app

# requirements for propnet
RUN apt-get update
RUN apt-get install graphviz libgraphviz-dev -y
RUN pip install numpy scipy pygraphviz
ADD requirements.txt /home/project/dash_app/requirements.txt
ADD setup.py /home/project/dash_app/setup.py
RUN pip install --no-cache-dir -r requirements.txt

ADD . /home/project/dash_app
RUN pip install -e .

# set up propnet env vars
ENV PROPNET_NUM_WORKERS=8
ENV PMG_MAPI_KEY='MATERIALS_PROJECT_KEY_HERE'
ENV PROPNET_CORRELATION_STORE_FILE="CORRELATION_STORE_FILE_OR_JSON_STRING"

# requirements for binder
RUN pip install --no-cache-dir notebook==5.*
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}


# start web server
EXPOSE 8000
CMD gunicorn --workers=$PROPNET_NUM_WORKERS --timeout=300 --bind=0.0.0.0 app:server
