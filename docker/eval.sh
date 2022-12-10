#!/bin/bash

source "$(dirname "$0")/_init.sh"

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
        --gpus '"device=0"' \
        --user $USER_ID:$GROUP_ID \
        carlasim/carla:0.9.13 \
        ./CarlaUE4.sh \
            -RenderOffScreen
    )

    echo "CARLA container ID: $CARLA_CONTAINER_ID"
    echo "Waiting for server to come online"
    sleep 15
else
    echo "Not starting CARLA"
fi

docker run \
    -it \
    --rm \
    --network container:$CARLA_CONTAINER_ID \
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
    "$DOCKER_IMAGE" \
    scripts/eval.sh "$@"

status=$?

if [ "$RUN_CARLA" -eq 1 ]; then
    echo "Stopping CARLA"
    docker stop "$CARLA_CONTAINER_ID"
    docker rm "$CARLA_CONTAINER_ID"
fi

exit $status
