#!/bin/bash

gpu=$1

CUDA_VISIBLE_DEVICES=${gpu} python eval.py \
    --data_name vidvrd \
    --cfg_path configs/vidvrd.yaml \
    --exp_dir experiments/vrdone_vidvrd \
    --eval_exp_dir \
    --eval_start_epoch 3 \
    --epochs 15 \
    --topk 1 \
    --eval_file_name eval_multi \

