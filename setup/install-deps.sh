#!/bin/bash

cd "$(dirname "$0")"

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please activate your virtual environment first."
    exit 1
fi

export CUDA_VERSION_SHORT=$(
    CUDA_PATH=$(realpath /usr/local/cuda)
    CUDA_NAME=$(basename $CUDA_PATH)
    echo $CUDA_NAME | sed -e 's/^cuda-/cu/' | tr -d '.'
)

export TORCH_VERSION=1.12.0

case $CUDA_VERSION_SHORT in
    cu113)
        export REQUIREMENTS_TORCH_OPTS="--extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION_SHORT}"
        export REQUIREMENTS_MMCV_OPTS="-f https://download.openmmlab.com/mmcv/dist/${CUDA_VERSION_SHORT}/torch${TORCH_VERSION}/index.html"
        export REQUIREMENTS_TORCH_SCATTER_OPTS="-f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION_SHORT}.html"
        ;;
    cu116)
        # todo: see if MMCV cu115 binaries are compatible
        export REQUIREMENTS_TORCH_OPTS="--extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION_SHORT}"
        export REQUIREMENTS_MMCV_OPTS=""
        export REQUIREMENTS_TORCH_SCATTER_OPTS="-f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION_SHORT}.html"
        ;;
    *)
        echo "Unsupported CUDA version: \"$CUDA_VERSION_SHORT\""
        exit 1
        ;;
esac

echo "Found CUDA version: \"$CUDA_VERSION_SHORT\""

export MAKEFLAGS="-j$(nproc)"

echo "Upgrading pip"
pip install --upgrade pip

for stage in 1 2 3 4; do
    echo "Installing stage $stage dependencies..."
    pip install --upgrade -r requirements-stage-$stage.txt
done

echo "Fixing opencv-python-headless"
pip uninstall -y opencv-python
pip install --force-reinstall opencv-python-headless
