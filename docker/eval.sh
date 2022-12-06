#!/bin/bash

set -euo pipefail

USER_ID=${SUDO_UID-$(id -u)}
GROUP_ID=${SUDO_GID-$(id -g)}
USER_NAME="$(id -un $USER_ID)"
IMAGE="$USER_NAME/transfuser"

TZ=${TZ:-$(cat /etc/timezone)}

if [ "$1" = "--carla" ]; then
    RUN_CARLA=1
    shift
else
    RUN_CARLA=0
fi

if [ "$RUN_CARLA" -eq 1 ]; then
    echo "Starting CARLA"
    CARLA_CONTAINER_ID=$(
        docker run \
        --detach \
        --rm \
        --gpus '"device=0"' \
        --network host \
        --user $USER_ID:$GROUP_ID \
        carlasim/carla:0.9.13 \
        ./CarlaUE4.sh \
            -RenderOffScreen
    )

    echo "Waiting for server to come online"
    sleep 15
else
    echo "Not starting CARLA"
fi

docker run \
    -it \
    --rm \
    --network host \
    --shm-size 512gb \
    --gpus '"device=1"' \
    -e TZ="$TZ" \
    -e PUID=$USER_ID \
    -e PGID=$GROUP_ID \
    -v "$(realpath models)":/models \
    -v "$(realpath results)":/results \
    -v "$(realpath dataset)":/dataset \
    -v "$(realpath logs)":/logs \
    -v "$(pwd)":/code/transfuser \
    -e CARLA_HOST=localhost \
    "$IMAGE" \
    scripts/eval.sh \
        $1 \
        $2

if [ "$RUN_CARLA" -eq 1 ]; then
    echo "Stopping CARLA"
    docker stop "$CARLA_CONTAINER_ID"
fi
