#!/bin/bash

cd "$(dirname "$0")/.."

echo "Training on a single GPU"

exec ./team_code_transfuser/train.py "$@"
