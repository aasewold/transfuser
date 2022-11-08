#!/bin/bash

ROOT=$(dirname "$0")/..

export ROUTES=${ROOT}/scenario_runner/srunner/data/routes_devtest.xml
export SCENARIOS=${ROOT}/scenario_runner/srunner/data/no_scenarios.json
export REPETITIONS=1
export DEBUG_CHALLENGE=1
export CHECKPOINT_ENDPOINT=${RESULTS_PATH}/test_run.json

# export TEAM_AGENT=${ROOT}/leaderboard/leaderboard/autoagents/human_agent.py
# export TEAM_AGENT=${ROOT}/team_code_autopilot/autopilot.py
export TEAM_AGENT=${ROOT}/team_code_transfuser/submission_agent.py
export TEAM_CONFIG=${MODELS_PATH}/transfuser-2022-10-26_14-00-47

# export CHALLENGE_TRACK_CODENAME=MAP
export CHALLENGE_TRACK_CODENAME=SENSORS

${ROOT}/leaderboard/scripts/run_evaluation.sh
