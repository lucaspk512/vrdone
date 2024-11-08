#!/bin/bash

gpu=$1

CUDA_VISIBLE_DEVICES=${gpu} python eval.py \
    --data_name vidvrd \
    --cfg_path configs/vidvrd.yaml \
    --exp_dir experiments/vrdone_vidvrd \
    --ckpt_path experiments/vrdone_vidvrd/ckpt_vidvrd.pth \
    --topk 8 \
