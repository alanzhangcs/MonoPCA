#!/bin/bash
# Usage: bash scripts/test.sh [GPUS] [PORT] [CONFIG]
# Set `tester.checkpoint_path` in the config to the checkpoint to evaluate.

GPUS=${1:-"0"}
PORT=${2:-29501}
CONFIG_PATH=${3:-"configs/monopca_kitti_nusc_waymo.yaml"}

NUM_GPUS=$(echo $GPUS | awk -F',' '{print NF}')
export CUDA_VISIBLE_DEVICES=$GPUS

echo "Using GPUs: $GPUS"
echo "Port: $PORT"
echo "Config: $CONFIG_PATH"

torchrun --nproc_per_node=$NUM_GPUS \
         --master_port=$PORT \
         tools/train_val.py \
         --config $CONFIG_PATH \
         -e
