#!/bin/bash

cd "$(dirname "$0")/.."

USER_ID=${SUDO_UID-$(id -u)}
GROUP_ID=${SUDO_GID-$(id -g)}
USER_NAME="$(id -un $USER_ID)"
IMAGE="$USER_NAME/transfuser"

TZ=${TZ:-$(cat /etc/timezone)}

if [ ! -d dataset ]; then
    echo "Missing dataset directory"
    if [ "$HOSTNAME" = "nap02" ]; then
        DATASET_PATH="/data/datasets/transfuser"
        echo "Creating symlink from $DATASET_PATH"
        ln -s "$DATASET_PATH" dataset
    elif [ "$HOSTNAME" = "vc2vm3" ]; then
        DATASET_PATH="/nap02data/datasets/transfuser"
        echo "Creating symlink from $DATASET_PATH"
        ln -s "$DATASET_PATH" dataset
    else
        echo "Please create a symlink to the dataset directory"
        exit 1
    fi
fi

docker run \
    -it \
    --rm \
    --network host \
    --shm-size 512gb \
    --gpus all \
    -e TZ="$TZ" \
    -e PUID=$USER_ID \
    -e PGID=$GROUP_ID \
    -v "$(realpath models)":/models \
    -v "$(realpath results)":/results \
    -v "$(realpath dataset)":/dataset \
    -v "$(realpath logs)":/logs \
    -v "$(pwd)":/code/transfuser \
    "$IMAGE" \
    "$@"
