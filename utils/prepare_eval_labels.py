import json
from VidVRD_helper.dataset import VidVRD, VidOR

def prepare_gts_for_vidvrd(config):
    dataset = VidVRD(**config['prepare_gt_config']['dataset_config'])
    indices = dataset.get_index(split="test")
    video_level_gts = dict()
    for vid in indices:
        video_level_gts[vid] = dataset.get_relation_insts(vid)
    
    if config['prepare_gt_config']['gt_relations_path'] is not None:
        print("saving ...")
        with open(config['prepare_gt_config']['gt_relations_path'], 'w') as f:
            json.dump(video_level_gts,f)
        print("done.")

def prepare_gts_for_vidor(config):

    dataset = VidOR(**config['prepare_gt_config']['dataset_config'])
    indices = dataset.get_index(split="validation")

    video_level_gts = dict()
    for vid in indices:
        video_level_gts[vid] = dataset.get_relation_insts(vid)
    
    if config['prepare_gt_config']['gt_relations_path'] is not None:
        print("saving ...")
        with open(config['prepare_gt_config']['gt_relations_path'], 'w') as f:
            json.dump(video_level_gts,f)
        print("done.")

