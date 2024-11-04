#!/bin/bash

gpu=$1

CUDA_VISIBLE_DEVICES=${gpu} python eval_vidor.py \
    --cfg_path configs/vidor_x.yaml \
    --exp_dir experiments/vrdone_vidor_x \
    --ckpt_path experiments/vrdone_vidor_x/ckpt_vidor_x.pth \
    --topk 4 \

