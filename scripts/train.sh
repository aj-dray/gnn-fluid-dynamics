#!/usr/bin/env bash


CONFIG_NAME=$1

## Get device from config file
CONFIG_PATH="config/$CONFIG_NAME.json"

# Use python to extract values from JSON instead of jq
DEVICE=$(../venv/bin/python3 -c "import json; print(json.load(open('$CONFIG_PATH')).get('settings', {}).get('device', ''))")
MACHINE=$(../venv/bin/python3 -c "import json; print(json.load(open('$CONFIG_PATH')).get('settings', {}).get('machine', ''))")
MULTI_GPU=$(../venv/bin/python3 -c "import json; print(json.load(open('$CONFIG_PATH')).get('settings', {}).get('multi_gpu', 'false'))")

## Check for debug tag
if [[ "$*" == *"--debug"* ]]; then
    DEBUG_TAG="--debug"
else
    DEBUG_TAG=""
fi

CONFIG="--config=$CONFIG_PATH"

## Check for GPU
if [[ $DEVICE == "cuda" && $MULTI_GPU == "True" ]] ; then
    # Use python to extract GPU_NUM value
    GPU_NUM=$(../venv/bin/python3 -c "import json; print(json.load(open('$CONFIG_PATH')).get('settings', {}).get('num_gpus', '1'))")
    echo "Using $GPU_NUM gpu(s)"
    ../venv/bin/torchrun --nproc_per_node=$GPU_NUM src/train.py $CONFIG $DEBUG_TAG # use torchrun for multi-gpu training
else
    echo ""
    echo "Machine: $MACHINE"
    echo "Device: $DEVICE"
    ../venv/bin/python3 src/train.py $CONFIG $DEBUG_TAG
fi
