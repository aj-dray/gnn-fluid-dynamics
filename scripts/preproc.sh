#!/bin/bash

FNAME=$1
CONFIG="config/$FNAME.json"

PYTHONPATH=src ../venv/bin/python3 src/preproc.py --config=$CONFIG
