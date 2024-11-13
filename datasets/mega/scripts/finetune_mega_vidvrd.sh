#!/bin/bash

n_gpu=$1
gpu_ids=$2

CUDA_VISIBLE_DEVICES=${gpu_ids} python -m torch.distributed.launch \
    --nproc_per_node=${n_gpu} \
    finetune_mega.py \
    --lr_rate 1 \
    --max_iter 210000 \
    --skip-test \
    --config-file configs/MEGA/vidvrd_R_101_C4_MEGA_1x_finetune.yaml \
    OUTPUT_DIR experiments/vidvrd/COCO21VRDfreq5_2gpu_finetune_lr1 \
