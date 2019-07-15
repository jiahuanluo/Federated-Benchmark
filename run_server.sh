#!/bin/bash
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

nohup python3 fl_server.py --config_file data/task_configs/${DATASET}/${MODEL}_task.json --port ${PORT} > fl_server.log &
