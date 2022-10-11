#!/usr/bin/env bash

cd "$(dirname "$0")"/..

TARGET_DIR=${MAPS_PATH:-maps}
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.13.tar.gz
tar -xf AdditionalMaps_0.9.13.tar.gz
rm AdditionalMaps_0.9.13.tar.gz
