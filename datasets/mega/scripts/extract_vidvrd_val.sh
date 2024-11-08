#!/bin/bash

gpu_id=$1

python -W ignore extract_test_features_vidvrd.py \
    --gpu_id ${gpu_id} \
    --lr_rate 1 \
    --max_iter 210000 \
    --config-file configs/MEGA/vidvrd_R_101_C4_MEGA_1x_finetune.yaml \
    OUTPUT_DIR experiments/vidvrd/COCO21VRDfreq5_2gpu_finetune_lr1 \
