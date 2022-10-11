FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    apt-get install -y \
        python3.8 python3.8-dev python3-venv python-is-python3 \
        pkg-config \
        libsm6 libxext6 libxrender1 \
        wget unzip

ENV VIRTUAL_ENV=/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY setup /tmp/setup
RUN /tmp/setup/install-deps.sh && \
    rm -rf /tmp/setup

ENV TRANSFUSER_PATH=/code/transfuser
RUN mkdir -p ${TRANSFUSER_PATH}
WORKDIR ${TRANSFUSER_PATH}

ENV PYTHONPATH="${TRANSFUSER_PATH}/scenario_runner:${TRANSFUSER_PATH}/leaderboard:${TRANSFUSER_PATH}/carla:${PYTHONPATH}"

ENV MODELS_PATH=/models
RUN mkdir ${MODELS_PATH}
VOLUME ${MODELS_PATH}

ENV DATASET_PATH=/dataset
RUN mkdir ${DATASET_PATH}
VOLUME ${DATASET_PATH}

ENV RESULTS_PATH=/results
RUN mkdir ${RESULTS_PATH}
VOLUME ${RESULTS_PATH}

COPY ./ ${TRANSFUSER_PATH}/
