#!/bin/bash

# 0 - 7000

gpu=$1

CUDA_VISIBLE_DEVICES=$gpu python -W ignore extract_gt_clip_features_vidor.py
