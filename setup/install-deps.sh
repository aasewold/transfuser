#!/bin/bash

cd "$(dirname "$0")"

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please activate your virtual environment first."
    exit 1
fi

echo "Upgrading pip"
pip install --upgrade pip

export PYTORCH_CUDA_INDEX=$(
    CUDA_PATH=$(realpath /usr/local/cuda)
    CUDA_NAME=$(basename $CUDA_PATH)
    echo $CUDA_NAME | sed -e 's/^cuda-/cu/' | tr -d .
)

case $PYTORCH_CUDA_INDEX in
    cu113)
        ;;
    cu116)
        ;;
    *)
        echo "Unsupported CUDA version: \"$PYTORCH_CUDA_INDEX\""
        exit 1
        ;;
esac

echo "Found CUDA version: \"$PYTORCH_CUDA_INDEX\""

export MAKEFLAGS="-j$(nproc)"

for stage in 1 2 3; do
    echo "Installing stage $stage dependencies..."
    pip install --upgrade -r requirements-stage-$stage.txt
done
