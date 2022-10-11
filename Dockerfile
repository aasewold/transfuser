FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    apt-get install -y \
        python3.8 python3.8-dev python3-venv python-is-python3 \
        pkg-config \
        libsm6 libxext6 libxrender1 \
        wget unzip

RUN mkdir -p /root/transfuser
WORKDIR /root/transfuser

ENV VIRTUAL_ENV=/root/transfuser/.venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY setup setup
RUN ./setup/install-deps.sh

ENV MODEL_PATH=/models
RUN mkdir ${MODEL_PATH}
VOLUME ${MODEL_PATH}

ENV DATASET_PATH=/dataset
RUN mkdir ${DATASET_PATH}
VOLUME ${DATASET_PATH}

ENV RESULTS_PATH=/results
RUN mkdir ${RESULTS_PATH}
VOLUME ${RESULTS_PATH}

COPY ./ ./
