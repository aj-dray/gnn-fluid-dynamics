CONFIG_NAME=$1
CONFIG="--config=config/$CONFIG_NAME.json"
../venv/bin/python3 src/rollout.py $CONFIG
