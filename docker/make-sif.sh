#!/bin/bash

source "$(dirname "$0")/_init.sh"

TARGETS=()

while [ $# -gt 0 ]; then
    case "$1" in
        --carla)
            shift
            TARGETS+=("docker://carlasim/carla:0.9.13")
            ;;
        --transfuser)
            shift
            TARGETS+=("docker-daemon://$DOCKER_IMAGE")
            ;;
        --carlo)
            shift
            CARLO_IMAGE="$1"
            shift
            TARGETS+=("docker-daemon://$CARLO_IMAGE")
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ "${#TARGETS[@]}" -eq 0 ]; then
    echo "No targets specified"
    echo "Specify one or more of --carla, --carlo, or --transfuser"
    exit 1
fi


if command -v apptainer > /dev/null; then
    APPTAINER=$(command -v apptainer)
elif command -v singularity > /dev/null; then
    APPTAINER=$(command -v singularity)
else
    echo "Could not find apptainer or singularity"
    exit 1
fi

echo "Found $APPTAINER"


# Make a sif file for each target
for TARGET in "${TARGETS[@]}"; do
    echo "Making sif file for $TARGET"
    $APPTAINER build transfuser.sif "$TARGET"
done
