#!/bin/sh

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please activate your virtual environment first."
    exit 1
fi

pip install wheel
pip install -r ./leaderboard/requirements.txt
pip install -r ./scenario_runner/requirements.txt
pip install -r ./team_code_transfuser/requirements.txt
pip install mmsegmentation==0.25.0 mmdet==2.25.0
