FROM ubuntu:latest

MAINTAINER Qhan <qhan@ailabs.tw>

## -----------------------------------------------------------------------------
## Install libraries

RUN apt-get update && apt-get install -y \

    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    htop \
    libgraphicsmagick1-dev \
    libatlas-dev \
    libavcodec-dev \
    libavformat-dev \
    libboost-all-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python-dev \
    python-numpy \
    python-protobuf \
    python3-pip \
    software-properties-common \
    tree \
    vim \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN cd /usr/local/bin \
 && ln -s /usr/bin/python3 python

RUN pip3 install \
    numpy \
    scipy \
    opencv-python

## dlib

RUN cd ~ && \
    mkdir -p dlib && \
    git clone https://github.com/davisking/dlib.git dlib/ && \
    cd dlib/ && \
    python setup.py install --yes USE_AVX_INSTRUCTIONS

## -----------------------------------------------------------------------------

WORKDIR /app

COPY .vimrc /app/
RUN mv .vimrc /root
