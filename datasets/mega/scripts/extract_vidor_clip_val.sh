#!/bin/bash

# 0 - 835

gpu=$1

CUDA_VISIBLE_DEVICES=$gpu python -W ignore extract_val_clip_features_vidor.py
