#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please activate your virtual environment first."
    exit 1
fi

export TORCH_VERSION=1.12.0
export CUDA_VERSION_SHORT=cu113

export REQUIREMENTS_MMCV_OPTS="-f https://download.openmmlab.com/mmcv/dist/${CUDA_VERSION_SHORT}/torch${TORCH_VERSION}/index.html"
export REQUIREMENTS_TORCH_SCATTER_OPTS="-f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION_SHORT}.html"

export MAKEFLAGS="-j$(nproc)"

echo
echo "Upgrading pip"
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

echo
echo "Fixing opencv-python-headless"
pip uninstall -y opencv-python
pip install --force-reinstall opencv-python-headless
