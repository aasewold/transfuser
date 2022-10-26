#!/bin/bash

cd "$(dirname "$0")/.."

echo "Training on multiple GPUs"

NVIDIA_GPUS="$(nvidia-smi -L)"
NUM_GPUS="$(echo "$NVIDIA_GPUS" | wc -l)"
CUDA_GPUS="$(echo "$NVIDIA_GPUS" | cut -d ' ' -f 2 | tr -d ':' | head -c -1 | tr '\n' ',')"
echo "Found ${NUM_GPUS} GPUs: ${CUDA_GPUS}"
echo "${NVIDIA_GPUS}"

export CUDA_VISIBLE_DEVICES="$CUDA_GPUS"
export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=1

exec torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --max_restarts=0 \
    --rdzv_id=123456780 \
    --rdzv_backend=c10d \
    team_code_transfuser/train.py "$@"
