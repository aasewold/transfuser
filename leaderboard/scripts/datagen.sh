
export SCENARIOS=${TRANSFUSER_PATH}/leaderboard/data/training/scenarios/Scenario10/Town10HD_Scenario10.json
export ROUTES=${TRANSFUSER_PATH}/leaderboard/data/training/routes/Scenario10/Town10HD_Scenario10.xml
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=MAP
export CHECKPOINT_ENDPOINT=${TRANSFUSER_PATH}/results/Town10HD_Scenario10.json
export SAVE_PATH=${TRANSFUSER_PATH}/results/Town10HD_Scenario10
export TEAM_AGENT=${TRANSFUSER_PATH}/team_code_autopilot/data_agent.py
export DEBUG_CHALLENGE=0
export RESUME=1
export DATAGEN=1

python3 ${TRANSFUSER_PATH}/leaderboard/leaderboard/leaderboard_evaluator_local.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--resume=${RESUME}
