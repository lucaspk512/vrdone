#!/bin/bash

gpu=$1

for i in {0..7}
do
    python -W ignore extract_gt_features_vidvrd.py --part_id $i --gpu_id $gpu
done

