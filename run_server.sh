#!/bin/bash
DATASET=$1
MODEL=$2
python3 federated_learning/fl_server.py data/task_configs/${DATASET}/${MODEL}_task.json
