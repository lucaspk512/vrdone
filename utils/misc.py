import os
import yaml
import numpy as np
import random
import logging
from copy import deepcopy

import torch
import torch.distributed as dist

class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed, disable_deterministic=False):
    """Set randon seed for pytorch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if disable_deterministic:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)

def create_folder(folder_path):
    dir_name = os.path.expanduser(folder_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, mode=0o777, exist_ok=True)

def save_config(config_dict, folder_path):
    with open(os.path.join(folder_path, 'config.yaml'), 'w') as file:
        file.write(yaml.dump(config_dict, indent=2, allow_unicode=True))

def reduce_loss(loss_dict):
    # reduce loss when distributed training, only for logging
    for loss_name, loss_value in loss_dict.items():
        loss_value = loss_value.data.clone()
        dist.all_reduce(loss_value.div_(dist.get_world_size()))
        loss_dict[loss_name] = loss_value
    return loss_dict


def exit_func(
        logger: logging.Logger, 
        model=None,
        model_ema=None,
        dataloader=None, 
        dataset=None
    ):
    logger.handlers.clear()
    logging.shutdown()

    if model is not None:
        del model
    if model_ema is not None:
        del model_ema
    if dataloader is not None:
        del dataloader
    if dataset is not None:
        del dataset

    print("exit done!")


def dict_to_device(data_dict: dict, device):
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            data_dict[k] = v.to(device)
        elif isinstance(v, str) or isinstance(v, int):
            pass
        elif isinstance(v, list):
            v_ = []
            for v_it in v:
                assert isinstance(v_it, torch.Tensor)
                v_.append(v_it.to(device))
            data_dict[k] = v_
        else:
            raise ValueError()
    return data_dict


def get_visual_features(box_features, tid, indices):
    visual_features = []
    keys = sorted(list(box_features.keys()))

    for duration in indices:
        start, end = duration[0], duration[1]
        features = []
        for k_ in keys:
            # k_ start from 1
            if (k_ - 1) < start: continue
            if (k_ - 1) >= end: break
            
            annos = box_features[k_]
            assert k_ == annos['frame_id']
            
            feature_id = np.where(annos['tids'] == tid)[0]
            assert len(feature_id) == 1
            features.append(annos['visual_features'][feature_id])
        features = torch.tensor(np.concatenate(features, axis=0))
        visual_features.append(features)
    
    return visual_features

def get_bboxes(trajectories, tid, indices):
    bboxes_with_durations = []
    for duration in indices:    
        start, end = duration[0], duration[1]
        bboxes = []
        for traj in trajectories[start: end]:
            for t_ in traj:
                if t_['tid'] == tid:
                    bbox = [
                        t_['bbox']['xmin'], t_['bbox']['ymin'], 
                        t_['bbox']['xmax'], t_['bbox']['ymax']
                    ]
                    bboxes.append(bbox)

        assert len(bboxes) == (end - start)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        bboxes_with_durations.append(bboxes)
        
    return bboxes_with_durations

def bbox_to_spatial_features(sbbox, obbox):
    assert isinstance(sbbox, torch.Tensor) and isinstance(obbox, torch.Tensor)
    s_ctx = (sbbox[:, 2] + sbbox[:, 0]) / 2
    s_cty = (sbbox[:, 3] + sbbox[:, 1]) / 2   
    
    s_w = sbbox[:, 2] - sbbox[:, 0]
    s_h = sbbox[:, 3] - sbbox[:, 1]
    
    o_ctx = (obbox[:, 2] + obbox[:, 0]) / 2
    o_cty = (obbox[:, 3] + obbox[:, 1]) / 2
    o_w = obbox[:, 2] - obbox[:, 0]
    o_h = obbox[:, 3] - obbox[:, 1]
    
    so_bbox_feat = torch.stack([
        (s_ctx - o_ctx) / o_ctx,
        (s_cty - o_cty) / o_cty,
        torch.log(s_w / o_w),
        torch.log(s_h / o_h),
        torch.log((s_w * s_h) / (o_w * o_h))
    ], dim=1)
    return so_bbox_feat


def entity_bbox_to_spatial_features(bboxes, w, h):
    assert isinstance(bboxes, torch.Tensor)

    bboxes_normed = deepcopy(bboxes)
    bboxes_normed[:, 0:4:2] /= w
    bboxes_normed[:, 1:4:2] /= h

    bbox_ctx = (bboxes_normed[:, 2] + bboxes_normed[:, 0]) / 2
    bbox_cty = (bboxes_normed[:, 3] + bboxes_normed[:, 1]) / 2
    bbox_w = bboxes_normed[:, 2] - bboxes_normed[:, 0]
    bbox_h = bboxes_normed[:, 3] - bboxes_normed[:, 1]

    diff_ctx = bbox_ctx[1:] - bbox_ctx[:-1]
    diff_cty = bbox_cty[1:] - bbox_cty[:-1]
    diff_w = bbox_w[1:] - bbox_w[:-1]
    diff_h = bbox_h[1:] - bbox_h[:-1]
    
    if len(diff_h) > 1:
        diff_ctx = torch.cat([torch.tensor([diff_ctx[0] - (diff_ctx[1] - diff_ctx[0])]), diff_ctx], dim=0)
        diff_cty = torch.cat([torch.tensor([diff_cty[0] - (diff_cty[1] - diff_cty[0])]), diff_cty], dim=0)
        diff_w = torch.cat([torch.tensor([diff_w[0] - (diff_w[1] - diff_w[0])]), diff_w], dim=0)
        diff_h = torch.cat([torch.tensor([diff_h[0] - (diff_h[1] - diff_h[0])]), diff_h], dim=0)
    else:
        diff_ctx = torch.cat([torch.tensor([diff_ctx[0]]), diff_ctx], dim=0)
        diff_cty = torch.cat([torch.tensor([diff_cty[0]]), diff_cty], dim=0)
        diff_w = torch.cat([torch.tensor([diff_w[0]]), diff_w], dim=0)
        diff_h = torch.cat([torch.tensor([diff_h[0]]), diff_h], dim=0)

    bbox_feat = [
        bbox_ctx, diff_ctx,
        bbox_cty, diff_cty,
        bbox_w, diff_w,
        bbox_h, diff_h,
    ]
    
    bbox_feat = torch.stack(bbox_feat, dim=1)
    return bbox_feat

def truncate_feats(
    so_feat, 
    pred,
    segment,
    max_seq_len,
    trunc_thresh=0.5,
    max_times=10,
):
    """
    Truncate feats and time stamps in a dict item

    data_dict = {
                 'so_feats'         : Tensor C x T
                 'segments'         : Tensor N x 2 (in feature grid)

    """
    # get the meta info
    feat_len = so_feat.shape[1]
    num_segs = segment.shape[0]

    # seq_len < max_seq_len
    if feat_len <= max_seq_len:
        return so_feat, pred, segment
    
    valid_seg = False
    for _ in range(max_times):
        st = random.randint(0, feat_len - max_seq_len)
        ed = st + max_seq_len
        window = torch.as_tensor([st, ed], dtype=torch.float32)

        # compute the intersection between the sampled window and all segments
        window = window[None].repeat(num_segs, 1)
        
        left = torch.maximum(window[:, 0], segment[:, 0])
        right = torch.minimum(window[:, 1], segment[:, 1])
        
        inter = (right - left).clamp(min=0)
        area_segs = torch.abs(segment[:, 1] - segment[:, 0])
        inter_ratio = inter / area_segs

        # only select those segments over the thresh
        seg_idx = (inter_ratio >= trunc_thresh)

        # with at least one action
        if seg_idx.sum().item() > 0:
            valid_seg = True; break

    if not valid_seg:
        return None
    
    so_feat = so_feat[:, st:ed]
    pred = pred[seg_idx]
    segment = torch.stack((left[seg_idx], right[seg_idx]), dim=1)
    segment = segment - st
    return so_feat, pred, segment 

