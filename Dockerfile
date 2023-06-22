FROM tensorflow/tensorflow:2.10.0-gpu

ARG DEBIAN_FRONTEND=noninteractive

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN apt-get update && \
    apt-get install -y git debconf-utils && \
    apt-get install -y vim less screen graphviz python3-tk wget && \
    apt-get install -y python3-venv
