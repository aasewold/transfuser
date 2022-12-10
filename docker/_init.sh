#/bin/bash

cd "$(dirname "$0")/.."

USER_ID=${SUDO_UID-$(id -u)}
GROUP_ID=${SUDO_GID-$(id -g)}
USER_NAME="$(id -un $USER_ID)"
DOCKER_IMAGE="$USER_NAME/transfuser"

TZ=${TZ:-$(cat /etc/timezone)}
