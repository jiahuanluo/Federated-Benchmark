#!/bin/bash
DATASET=$1
NUM_CLIENT=$2
MODEL=$3
echo ${NUM_CLIENT}
for i in $(seq 1 ${NUM_CLIENT}); do
    nohup python3 fl_client.py --gpu $(($i % 8)) --config_file data/task_configs/${DATASET}/${MODEL}_task_$i.json --ignore_load True &
done
