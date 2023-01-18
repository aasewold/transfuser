#!/bin/bash

ROOT=$(dirname "$0")/..

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 [dev|longest6] [model_path]"
    exit 1
fi

if [ "$1" = "dev" ]; then
    export ROUTES=${ROOT}/scenario_runner/srunner/data/routes_devtest.xml
    export SCENARIOS=${ROOT}/scenario_runner/srunner/data/no_scenarios.json
    RUNNER=leaderboard_evaluator
elif [ "$1" = "longest6" ]; then
    export ROUTES=${ROOT}/leaderboard/data/longest6/longest6.xml
    export SCENARIOS=${ROOT}/leaderboard/data/longest6/eval_scenarios.json
    RUNNER=leaderboard_evaluator_local
else
    echo "Unknown track: $1"
    exit 1
fi

MODEL_PATH="$2"
RESULTS_PATH="$MODEL_PATH/results/$(date +%Y%m%d_%H%M%S).json"
mkdir -p "$(dirname "$RESULTS_PATH")"

export TEAM_AGENT=${ROOT}/team_code_transfuser/agent.py
export TEAM_CONFIG=${MODEL_PATH}

export DATAGEN=0
export REPETITIONS=1
export DEBUG_CHALLENGE=0

python3 "${ROOT}/leaderboard/leaderboard/$RUNNER.py" \
    --routes=${ROUTES} \
    --scenarios=${SCENARIOS}  \
    --repetitions=${REPETITIONS} \
    --track=SENSORS \
    --checkpoint=${RESULTS_PATH} \
    --agent=${TEAM_AGENT} \
    --agent-config=${TEAM_CONFIG} \
    --debug=${DEBUG_CHALLENGE} \
    --host=${CARLA_HOST:-localhost}
