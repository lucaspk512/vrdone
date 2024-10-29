#!/bin/bash

gpu=$1

for i in {0..699}
do
    python -W ignore extract_gt_features_vidor.py \
        --part_id $i --gpu_id $gpu \
        --config_file configs/MEGA/partxx/VidORtrain_freq1_part01.yaml \
        
done
