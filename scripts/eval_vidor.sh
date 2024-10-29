#!/bin/bash

gpu=$1

# CUDA_VISIBLE_DEVICES=${gpu} python eval_vidor.py \
#     --cfg_path configs/vidor.yaml \
#     --exp_dir experiments/vrdone_vidor \
#     --eval_exp_dir \
#     --eval_start_epoch 3 \
#     --epochs 12 \
#     --topk 1 \

CUDA_VISIBLE_DEVICES=${gpu} python eval_vidor.py \
    --cfg_path configs/vidor.yaml \
    --exp_dir experiments/vrdone_vidor \
    --ckpt_path ckpts/ckpt_vidor.pth \
    --topk 1 \

