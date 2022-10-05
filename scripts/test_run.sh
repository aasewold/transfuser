#!/bin/bash

ROOT=$(dirname "$0")/..

export ROUTES=${ROOT}/scenario_runner/srunner/data/routes_devtest.xml
export REPETITIONS=1
export DEBUG_CHALLENGE=1
export TEAM_AGENT=${ROOT}/leaderboard/leaderboard/autoagents/human_agent.py
export CHECKPOINT_ENDPOINT=${ROOT}/results/test_run.json
export CHALLENGE_TRACK_CODENAME=SENSORS

${ROOT}/leaderboard/scripts/run_evaluation.sh
