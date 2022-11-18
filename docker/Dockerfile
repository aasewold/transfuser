FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    apt-get install -y \
        tzdata \
        python3.8 \
        python3.8-dev \
        python3.8-venv \
        python3.8-distutils \
        pkg-config \
        libsm6 libxext6 libxrender1 \
        wget unzip git

ENV VIRTUAL_ENV=/venv
RUN python3.8 -m venv --system-site-packages $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY setup /tmp/setup
RUN --mount=type=cache,target=/root/.cache/pip \
    /tmp/setup/install-deps.sh && \
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

COPY ./docker/docker-entry.sh /root/docker-entry.sh

ENTRYPOINT [ "/root/docker-entry.sh" ]
CMD [ "/bin/bash" ]

COPY ./ ${TRANSFUSER_PATH}/
