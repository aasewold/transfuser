#!/usr/bin/bash

filename=$(date +'%Y-%m-%d_%H-%M-%S.json')

n=0
until [ "$n" -ge 36 ]
do
    ./docker/eval.sh \
        --carla \
        longest6 \
        /models/pretrained/transfuser \
        models/pretrained/transfuser/results/$filename \
        && break;
    n=$((n+1))
done
