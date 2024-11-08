#!/bin/bash

n_gpu=$1
gpu_ids=$2

CUDA_VISIBLE_DEVICES=${gpu_ids} python -m torch.distributed.launch \
    --nproc_per_node=${n_gpu} \
    train_mega.py \
    --config-file configs/MEGA/vidvrd_R_101_C4_MEGA_1x_2gpu_freq5.yaml \
    OUTPUT_DIR experiments/vidvrd/COCO21VRDfreq5_2gpu \
