#!/bin/bash

gpu=$1

CUDA_VISIBLE_DEVICES=${gpu} python eval.py \
    --data_name vidor \
    --cfg_path configs/vidor_x.yaml \
    --exp_dir experiments/vrdone_vidor_x \
    --eval_exp_dir \
    --eval_start_epoch 3 \
    --epochs 12 \
    --topk 1 \
    --eval_file_name eval_multi \

