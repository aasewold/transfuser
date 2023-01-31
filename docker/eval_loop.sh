#!/usr/bin/bash

n=0
until [ "$n" -ge 36 ]
do
    ./docker/eval.sh \
        --carla \
        longest6 \
        /models/pretrained/transfuser \
        models/pretrained/transfuser/results/20221207_153000.json \
        && break;
    n=$((n+1))
done
