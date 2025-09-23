#!/bin/bash

# Set the path and prefix
HF_OWNER="neashton"
HF_PREFIX="ahmedml"

# Set the local directory to download the files
LOCAL_DIR="/home/ajd246/rds/rds-t2-cs181-iLDMbuOsGy8/ajd246" # TODO

# Create the local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Loop through the run folders from 1 to 500
for i in $(seq 1 50); do
    RUN_DIR="run_$i"
    RUN_LOCAL_DIR="$LOCAL_DIR/$RUN_DIR"

    # Create the run directory if it doesn't exist
    mkdir -p "$RUN_LOCAL_DIR"

    # Download the slices file
    wget "https://huggingface.co/datasets/${HF_OWNER}/${HF_PREFIX}/resolve/main/$RUN_DIR/slices/" -O "$RUN_LOCAL_DIR/slices/"

done
