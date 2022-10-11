#!/usr/bin/env bash

cd "$(dirname "$0")"/..

TARGET_DIR=${DATASET_PATH:-data}
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# Download license file
wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/data/LICENSE.txt

# Download 2022 dataset
for scenario in left ll lr right rl rr s1 s3 s4 s7 s8 s9 s10
do
	wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/data/2022_data/${scenario}.zip
	unzip -q ${scenario}.zip
	rm ${scenario}.zip
done
