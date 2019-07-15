#!/bin/bash
DATASET=$1
MODEL=$2
nohup python3 fl_server.py --config_file data/task_configs/${DATASET}/${MODEL}_task.json --port 1234
