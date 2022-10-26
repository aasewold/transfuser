#!/bin/bash

cd "$(dirname "$0")/.."

USER_ID=${SUDO_UID-$(id -u)}
GROUP_ID=${SUDO_GID-$(id -g)}
USER_NAME="$(id -un $USER_ID)"
IMAGE="$USER_NAME/transfuser"

docker run \
    -it \
    --rm \
    --network host \
    --shm-size 512gb \
    --gpus all \
    -e PUID=$USER_ID \
    -e PGID=$GROUP_ID \
    -v "$(pwd)/models":/models \
    -v "$(pwd)/results":/results \
    -v "$(pwd)/dataset":/dataset \
    -v "$(pwd)":/code/transfuser \
    "$IMAGE"
