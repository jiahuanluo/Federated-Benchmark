#!/bin/bash
DATASET=$1
python3 federated_learning/fl_server.py data/task_configs/${DATASET}/faster_rcnn_task.json
