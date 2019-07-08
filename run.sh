#!/bin/bash
DATASET=$1
NUM_CLIENT=$2
echo ${NUM_CLIENT}
for i in $(seq 1 ${NUM_CLIENT}); do
    nohup python3 federated_learning/fl_client.py $(($i % 8)) data/task_configs/${DATASET}/faster_rcnn_task$i.json True &
done
