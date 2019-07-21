#!/bin/bash

set -x
set -e

DATASET=$1
MODEL=$2
PORT=$3

if [ ! -n "$DATASET" ];then
	echo "Please input dataset"
	exit
fi

if [ ! -n "$MODEL" ];then
        echo "Please input model name"
        exit
fi

if [ ! -n "$PORT" ];then
        echo "please input server port"
        exit
fi
LOG="experiments/logs/`date +'%m-%d'`/fl_server.log"
echo Loggin output to "$LOG"

nohup python3 fl_server.py --config_file data/task_configs/${DATASET}/${MODEL}_task.json --port ${PORT} > ${LOG} &
