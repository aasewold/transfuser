#!/bin/bash

source "$(dirname "$0")/_init.sh"

docker buildx build \
    -f docker/Dockerfile \
    -t "$DOCKER_IMAGE" \
    .
