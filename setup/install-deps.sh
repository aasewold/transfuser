#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please activate your virtual environment first."
    exit 1
fi

export MAKEFLAGS="-j$(nproc)"

echo
echo "Upgrading pip"
pip install --upgrade pip wheel setuptools

echo
echo "Installing requirements"
pip install -r requirements.txt

echo
echo "Fixing opencv-python-headless"
pip uninstall -y opencv-python
pip install --force-reinstall opencv-python-headless
