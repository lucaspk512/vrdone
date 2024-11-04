#!/bin/bash

num_gpu=$1
gpu=$2

CUDA_VISIBLE_DEVICES=${gpu} torchrun --nnodes=1 --nproc_per_node=${num_gpu} \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train_vidor.py \
    --cfg_path configs/vidor.yaml \
    --exp_dir experiments/vrdone_vidor \
