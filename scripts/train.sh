#!/bin/bash
# Usage: bash scripts/train.sh [GPUS] [PORT] [CONFIG] [SEED]
# Example: bash scripts/train.sh 0,1,2,3 29500 configs/monopca_kitti_nusc_waymo.yaml 444

GPUS=${1:-"0,1,2,3"}
PORT=${2:-29500}
CONFIG_PATH=${3:-"configs/monopca_kitti_nusc_waymo.yaml"}
SEED=${4:-444}

NUM_GPUS=$(echo $GPUS | awk -F',' '{print NF}')
export CUDA_VISIBLE_DEVICES=$GPUS

echo "Using GPUs: $GPUS"
echo "Port: $PORT"
echo "Config: $CONFIG_PATH"
echo "Seed: $SEED"

torchrun --nproc_per_node=$NUM_GPUS \
         --master_port=$PORT \
         tools/train_val.py \
         --config $CONFIG_PATH \
         --seed $SEED
