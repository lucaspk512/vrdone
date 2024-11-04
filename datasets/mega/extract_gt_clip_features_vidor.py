#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import time
import yaml
from tqdm import tqdm
import pickle
from copy import deepcopy
from PIL import Image
from collections import defaultdict
import argparse

import torch
import torch.utils.data as data

import clip

def get_video_annos_files(anno_path, subsets):
    vid_to_path = {}
    group_ids = set()
    
    for sb in subsets:
        group_ids.update(list(os.listdir(os.path.join(anno_path, sb))))
    
    group_ids = sorted(list(group_ids))
    
    for gid in group_ids:
        for sb in subsets:
            if os.path.exists(os.path.join(anno_path, sb, gid)):
                vname_files = sorted(os.listdir(os.path.join(anno_path, sb, gid)))
                for vn_f in vname_files:
                    vn = vn_f.strip().split('.')[0]
                    vid_to_path[gid + '_' + vn] = os.path.join(anno_path, sb, gid, vn_f)

    print('Number of {} videos: {}'.format(subsets, len(list(vid_to_path.keys()))))
    return vid_to_path


def load_annos(vid_to_path):
    print('----------------- load annotations ------------------')

    v_anno = dict()
    for vid in tqdm(list(vid_to_path.keys())):
        # 'version', 'video_id', 'video_hash', 'video_path', 'frame_count', 
        # 'fps', 'width', 'height', 'subject/objects', 'trajectories', 'relation_instances'
        
        with open(vid_to_path[vid], 'r') as f:
            anno = json.load(f)
        
        width, height = anno['width'], anno['height']
        
        vid_tid_to_category = dict()
        for so in anno['subject/objects']:
            so_category = ' '.join(so['category'].strip().split('_'))
            vid_tid_to_category[so['tid']] = {
                "category": so_category,
                "trajectories": []
            }

        for fid in range(len(anno['trajectories'])):
            for traj_anno in anno['trajectories'][fid]:
                x_min, y_min = traj_anno['bbox']['xmin'], traj_anno['bbox']['ymin']
                x_max, y_max = traj_anno['bbox']['xmax'], traj_anno['bbox']['ymax']
                
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(x_max, width-1), min(y_max, height-1)
                assert x_min < x_max and y_min < y_max
                
                vid_tid_to_category[traj_anno['tid']]['trajectories'].append([x_min, y_min, x_max, y_max])
            
            for tid in vid_tid_to_category.keys():
                if len(vid_tid_to_category[tid]['trajectories']) != fid + 1:
                    vid_tid_to_category[tid]['trajectories'].append([-1, -1, -1, -1])

        # relations = []
        # for rid in range(len(anno['relation_instances'])):
        #     pred = ' '.join(anno['relation_instances'][rid]['predicate'].strip().split('_'))
        #     rlt = anno['relation_instances'][rid]
        #     rlt['predicate'] = pred
        #     relations.append(rlt)
        
        v_anno[vid] = {
            "traj": vid_tid_to_category,
            # "relation": relations,
            "frame_count": len(anno['trajectories']),
            "width": anno['width'],
            "height": anno['height'],
        }
        
    print('----------------- done ------------------')
    return v_anno

def video_clip_collate_fn(batch_data):
    imgs = []
    masks = []
    indexes = [bd[0] for bd in batch_data]
    
    for bid in range(len(batch_data)):
        imgs += batch_data[bid][1]
        masks += batch_data[bid][2]
    return indexes, imgs, masks 

class VideoClip(data.Dataset):
    def __init__(self, vid, frame_count, traj, size):
        self.frame_dir = config['vidor']['frame_path']
        self.vid = vid
        self.traj = traj
        self.frame_count = frame_count
        gid, vn = self.vid.split('_')
        self.frame_names = sorted(list(os.listdir(os.path.join(self.frame_dir, gid, vn))))
        assert len(self.frame_names) >= self.frame_count
        self.tids = sorted(list(self.traj.keys()))
        self.H, self.W = size

    def __len__(self):
        return self.frame_count

    def __getitem__(self, index):
        all_images = []
        # drop mask means if the feature will be kept
        drop_masks = []
        gid, vn = self.vid.split('_')
        frame_path = os.path.join(self.frame_dir, gid, vn, self.frame_names[index])
        frame_img = Image.open(frame_path)
        all_images.append(frame_img)
        drop_masks.append(False)
        for k in self.tids:
            if self.traj[k]['trajectories'][index] == [-1, -1, -1, -1]:
                all_images.append(frame_img.crop((0, 0, self.W, self.H)))
                drop_masks.append(True)
            else:
                # There may be some negative coords in the bbox, Image will pad them with 0.
                all_images.append(frame_img.crop(tuple(self.traj[k]['trajectories'][index])))
                drop_masks.append(False)
        return index, all_images, drop_masks


if __name__ == "__main__":
    ## args
    parser = argparse.ArgumentParser("Extract CLIP features demo", add_help=True)
    parser.add_argument("--is_part", action="store_true", default=False)
    parser.add_argument("--start_video", type=int)
    parser.add_argument("--end_video", type=int)
    parser.add_argument("--config", type=str, default="./configs/CLIP/default.yaml")
    args = parser.parse_args()

    ## config
    config_file = args.config
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    anno_path = config['vidor']['annotation_path']
    
    # subsets = ["training", "validation"]
    subsets = ["training"]
    vid_to_path = get_video_annos_files(anno_path, subsets)
    
    video_ids = sorted(list(vid_to_path.keys()))
    
    if args.is_part:
        start_video_idx = args.start_video
        end_video_idx = args.end_video
        end_video_idx = min(len(video_ids), end_video_idx)
        video_ids = video_ids[start_video_idx: end_video_idx]

    vid_to_path_filtered = {k: vid_to_path[k] for k in video_ids}
    
    ## annos
    annos = load_annos(vid_to_path_filtered)
    
    ## model
    device = config['device']
    model, preprocess = clip.load("ViT-B/32", device=device)

    output_dir = config['clip']['output_path_vidor_train']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    bsz = config['clip']['batch_size']
    nw = config['clip']['num_workers']

    num_videos = len(video_ids)

    ################## clip feature ###################
    start_time = time.time()
    for video_idx, vid in enumerate(video_ids):
        
        video_start_time = time.time()
        print("processing the {}th / {} video".format(video_idx+1, num_videos))

        v_anno = deepcopy(annos[vid])
        if os.path.exists(os.path.join(output_dir, vid+'.pkl')):
            continue
        
        vdataset = VideoClip(vid, v_anno['frame_count'], v_anno['traj'], [v_anno['height'], v_anno['width']])
        vdataloader = data.DataLoader(
            vdataset, 
            batch_size=bsz, 
            shuffle=False, 
            num_workers=nw, 
            drop_last=False, 
            persistent_workers=(True if nw > 0 else False),
            collate_fn=video_clip_collate_fn
        )
        
        len_tid = len(vdataset.tids) + 1 # include full image
        video_features = defaultdict(list)
        video_drop_mask = defaultdict(list)
        for vdata in tqdm(vdataloader):
            indexes, imgs, drop_masks = vdata

            processed_imgs = []
            for img in imgs:
                processed_imgs.append(preprocess(img).unsqueeze(0))
            processed_imgs = torch.cat(processed_imgs, dim=0).to(device)
        
            with torch.no_grad():
                img_features = model.encode_image(processed_imgs)
            
            assert img_features.shape[0] % len_tid == 0
            
            # -1 to note full image
            for tid_idx, tid in zip(range(len_tid), ['full'] + vdataset.tids):
                video_features[tid].append(img_features[tid_idx::len_tid])
                assert video_features[tid][-1].ndim == 2
                
                video_drop_mask[tid].append(torch.tensor(drop_masks[tid_idx::len_tid]).to(device))
                assert video_drop_mask[tid][-1].ndim == 1
            
        for tid in video_features.keys():
            video_features[tid] = torch.cat(video_features[tid], dim=0)
            video_drop_mask[tid] = torch.cat(video_drop_mask[tid], dim=0)
            
            video_features[tid][video_drop_mask[tid], :] = 0
        
        features = {
            "full_image": video_features['full'].cpu().numpy()
        }

        for tid in vdataset.tids:
            features[tid] = video_features[tid].cpu().numpy()
        
        with open(os.path.join(output_dir, vid+'.pkl'), 'wb') as of:
            pickle.dump(features, of)

        print("finished for {:.4f}s, each frame is {:.4f}s".format(time.time() - video_start_time, (time.time() - video_start_time) / len(vdataset)))
    
    print("total time is {:.4f}s".format(time.time() - start_time))
    #################################################