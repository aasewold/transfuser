#!/bin/bash

cd "$(dirname "$0")"/..

TARGET_DIR=${MODELS_PATH:-model_ckpt}
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/models_2022.zip
unzip models_2022.zip
rm models_2022.zip
