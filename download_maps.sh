#!/usr/bin/env bash

# Download and install additional Carla maps
mkdir maps
cd maps
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.13.tar.gz
tar -xf AdditionalMaps_0.9.13.tar.gz
rm AdditionalMaps_0.9.13.tar.gz
cd ..