#!/bin/bash

gpu=$1

CUDA_VISIBLE_DEVICES=${gpu} python eval_vidor.py \
    --cfg_path configs/vidor.yaml \
    --exp_dir experiments/vrdone_vidor \
    --ckpt_path experiments/vrdone_vidor/ckpt_vidor.pth \
    --topk 6 \

