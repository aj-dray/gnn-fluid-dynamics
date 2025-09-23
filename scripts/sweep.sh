#!/bin/bash

# Hyperparameter sweep wrapper script
# Usage: ./scripts/sweep.sh <sweep_config> [array_id] [array_total]

# set -x

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Get arguments
SWEEP_CONFIG="../project/config/$1.json"
ARRAY_ID="${2:-${SLURM_ARRAY_TASK_ID:-0}}"
ARRAY_TOTAL="${3:-${SLURM_ARRAY_TASK_COUNT:-1}}"

# Run the sweep
../venv/bin/python3 src/sweep.py "$SWEEP_CONFIG" --array_id "$ARRAY_ID" --array_total "$ARRAY_TOTAL" 2>&1
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Sweep failed with exit code $EXIT_CODE"
fi
