import os
import json
import pickle
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

import torch
import torch.utils.data as data

from .category import vidor_category_name_to_id, vidor_category_id_to_name, vidor_pred_name_to_id, vidor_pred_id_to_name
import utils.misc as utils

TO_REMOVE = 1

class VidOR(data.Dataset):
    def __init__(self, config, scale=None):
        self.split = config['split']
        assert self.split in ['training', 'validation']

        ## not used
        self.min_frames_th = config['min_frames_th']
        self.max_proposal = config['max_proposal']
        self.max_preds = config['max_preds']
        self.score_th = config['score_th']

        ## feat dim
        self.dim_visualfeature = config['dim_visualfeature']
        self.dim_wordfeature = config['dim_wordfeature']

        ## dirs
        self.anno_dir = config['ann_dir']
        self.info_dir = config['info_dir']
        self.cache_tag = config['cache_tag']
        self.cache_dir = config['cache_dir']
        self.policy_path = config['policy_path']
        self.gt_boxfeatures_dir = config['gt_boxfeatures_dir']

        ## config
        self.feat_stride = config['feat_stride']
        self.max_seq_len = config['max_seq_len']

        ## name to id
        self.entity_cat_name_to_id = vidor_category_name_to_id
        self.pred_cat_name_to_id = vidor_pred_name_to_id
        self.entity_cat_id_to_name = vidor_category_id_to_name
        self.pred_cat_id_to_name = vidor_pred_id_to_name

        ## anno dir and video name
        self.video_ann_dir = os.path.join(self.anno_dir, self.split)
        self.video_name_list = self._prepare_video_names()

        ## if scale is True, then use a small part data for debug
        self.scale = scale
        if scale:
            self.video_name_list = self.video_name_list[:scale]
        
        ## build cache
        cache_path = (self.cache_tag + "_" + "VidOR_{}".format(self.split))

        ## add config specific to training set
        if self.split == "training":
            self.cut_max_preds = config['cut_max_preds']
            self.proposal_max_preds = config['proposal_max_preds']
            self.num_pairs = config['num_pairs']
            self.video_num_pairs = []
        else:
            self.proposal_min_frames = config['proposal_min_frames']
            self.index_shift = config['shift']
            self.random_stride = config['random_stride']
            self.stride_offset = config['stride_offset']
            assert self.proposal_min_frames > self.stride_offset
            if not self.index_shift:
                cache_path += '_wo_shift'

        self.cache_path = os.path.join(self.cache_dir, cache_path)
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        
        ## (key step!) process data
        self.process_data()
        ## done

    def _prepare_video_names(self):
        group_list = os.listdir(self.video_ann_dir)
        group_list = sorted(group_list)
        video_name_list = []
        for group_name in group_list:
            video_list = os.listdir(os.path.join(self.video_ann_dir, group_name))
            video_list = sorted(video_list)
            video_list = [group_name + "_" + v.split('.')[0] for v in video_list]
            video_name_list += video_list
        return video_name_list
    
    def process_data(self):
        """
        not very elegant
        """
        print("Processing data into cache files at cache path: {}".format(self.cache_path))
        if self.split == 'training':
            ## generate cache
            for video_name in tqdm(self.video_name_list):
                if (not os.path.exists(os.path.join(self.cache_path, video_name+'.pkl'))):
                    data_ = self._prepare_cache(video_name)
                    with open(os.path.join(self.cache_path, video_name+'.pkl'), 'wb') as f:
                        pickle.dump(data_, f)

            ## calculate the number of pairs for each video
            if not os.path.exists(self.policy_path):
                if self.scale:
                    raise ValueError("Remove `scale` and use the whole dataset to generate the policy application file!!!")

                video_np = []
                for video_name in tqdm(self.video_name_list):
                    with open(os.path.join(self.cache_path, video_name+'.pkl'), 'rb') as f:
                        data_ = pickle.load(f)
                    
                    if len(data_.keys()) != 0:
                        video_np.append([video_name, len(data_['relation_keys'])])

                with open(self.policy_path, 'w') as pf:
                    for vnp in video_np:
                        pf.write(" ".join([str(_it) for _it in vnp]) + "\n")
                self.video_num_pairs = deepcopy(video_np)
            else:
                with open(self.policy_path, 'r') as pf:
                    video_np = pf.readlines()

                for vnp in video_np:
                    data_np = vnp.strip().split(" ")
                    assert len(data_np) == 2

                    if data_np[0] not in self.video_name_list:
                        break
                    self.video_num_pairs.append([data_np[0], int(data_np[1])])

            ## apply policy  
            print("Applying policy...")        
            self.apply_policy()
            print("Applying is done.")
            
        else:
            self.video_features = dict()
            
            ## generate cache and load feauture
            for video_name in tqdm(self.video_name_list):
                if (not os.path.exists(os.path.join(self.cache_path, video_name+'.pkl'))):
                    data_ = self._prepare_cache(video_name)
                    with open(os.path.join(self.cache_path, video_name+'.pkl'), 'wb') as f:
                        pickle.dump(data_, f)
                else:
                    with open(os.path.join(self.cache_path, video_name+'.pkl'), 'rb') as f:
                        data_ = pickle.load(f)
                self.video_features[video_name] = data_

        print("Process done.")

    def apply_policy(self):
        """
        build a policy for loading data with balance
        """
        self.policy = list()
        
        current_num_pairs = 0
        policy_index = 0
        self.policy.append([])
        
        for pair_info in self.video_num_pairs:
            video_name, len_video_pairs = pair_info[0], pair_info[1]
            
            if len_video_pairs + current_num_pairs < self.num_pairs:
                ## if not too many, direct add
                self.policy[policy_index].append([video_name, (0, len_video_pairs)])
                current_num_pairs += len_video_pairs
            else:
                ## else, add to the next step 
                start_index = 0
                while len_video_pairs + current_num_pairs >= self.num_pairs:
                    self.policy[policy_index].append([video_name, (start_index, start_index + self.num_pairs - current_num_pairs)])
                    len_video_pairs -= (self.num_pairs - current_num_pairs)
                    start_index += (self.num_pairs - current_num_pairs)

                    current_num_pairs = 0
                    policy_index += 1
                    self.policy.append([])
                
                if len_video_pairs > 0:
                    assert len_video_pairs + current_num_pairs < self.num_pairs
                    self.policy[policy_index].append([video_name, (start_index, start_index + len_video_pairs)])
                    current_num_pairs += len_video_pairs
        

    def _prepare_cache(self, video_name):
        if self.split == "training":
            return self._prepare_train(video_name)
        else:
            return self._prepare_val(video_name)
    
    def _prepare_train(self, video_name):
        assert self.split == 'training'
        
        ## 1. get anno
        group_id, video_id = tuple(video_name.split('_'))
        anno_path = os.path.join(self.video_ann_dir, group_id, video_id+'.json')
        with open(anno_path, 'r') as gf:
            video_anno = json.load(gf)
    
        ## 2. return if info of the video is invalid.
        if len(video_anno['relation_instances']) == 0:
            return {}

        ## 3. get gt features
        with open(os.path.join(self.gt_boxfeatures_dir, video_name+'.pkl'), 'rb') as gf:
            gt_box_features = pickle.load(gf)
        
        ## 4. get valid frames
        traj_frames = defaultdict(list)
        for frame_id, frame_anno in enumerate(video_anno['trajectories']):
            for bbox_anno in frame_anno:
                traj_frames[bbox_anno['tid']].append(frame_id) 

        ## 5. tid to number_id 
        tids = sorted(list(traj_frames.keys()))
        tid_to_index = {t1: t2 for t1, t2 in zip(tids, range(len(tids)))}

        ## 6. get features
        visual_features = dict()
        entity_bboxes = dict()
        entity_classes = dict()
        traj_intervals = dict()

        for tid in tids:
            index  = tid_to_index[tid]
            
            ## 6.1 get interval
            traj_frames[tid] = sorted(traj_frames[tid])
            frame_indices = torch.tensor(deepcopy(traj_frames[tid]))
            diff = frame_indices[1:] - frame_indices[:-1]

            ## split interval
            start_indices = torch.cat((torch.tensor([0]), torch.nonzero(diff > 1).squeeze(1) + 1))
            end_indices = torch.cat((torch.nonzero(diff > 1).squeeze(1), torch.tensor([len(frame_indices) - 1])))
            start_indices, end_indices = frame_indices[start_indices], frame_indices[end_indices] + 1
            indices_ = torch.stack([start_indices, end_indices], dim=-1)
            traj_intervals[index] = indices_

            indices = indices_.tolist()
            ## 6.2 visual features
            visual_features[index] = utils.get_visual_features(gt_box_features, tid, indices)
            entity_bboxes[index] = utils.get_bboxes(video_anno['trajectories'], tid, indices)
            assert len(visual_features[index]) == len(indices) and len(entity_bboxes[index]) == len(indices)

        ## 7. entity classes
        for so in video_anno['subject/objects']:
            index  = tid_to_index[so['tid']]
            entity_classes[index] = self.entity_cat_name_to_id[so['category']]

        ## 8. process relation
        relation_merged = defaultdict(list)
        relation_keys = set()

        for relation_anno in video_anno['relation_instances']:
            sub_index, obj_index = tid_to_index[relation_anno['subject_tid']], tid_to_index[relation_anno['object_tid']]
            bf, ef = relation_anno['begin_fid'], relation_anno['end_fid']
            sub_valid = (traj_intervals[sub_index][:, 0] <= bf) & (traj_intervals[sub_index][:, 1] >= ef)
            obj_valid = (traj_intervals[obj_index][:, 0] <= bf) & (traj_intervals[obj_index][:, 1] >= ef)

            assert torch.sum(sub_valid) == 1 and torch.sum(obj_valid) == 1

            sub_interval_index = torch.nonzero(sub_valid).squeeze(1)[0].item()
            obj_interval_index = torch.nonzero(obj_valid).squeeze(1)[0].item()

            sub_interval = traj_intervals[sub_index][sub_interval_index]
            obj_interval = traj_intervals[obj_index][obj_interval_index]

            so_start, so_end = max(sub_interval[0], obj_interval[0]), min(sub_interval[1], obj_interval[1])
            assert so_start < so_end

            relation_merged[(sub_index, obj_index, sub_interval_index, obj_interval_index)].append({
                "predicate": self.pred_cat_name_to_id[relation_anno['predicate']],
                "begin_fid": bf, 
                "end_fid": ef,
            })
            
            relation_keys.add((sub_index, obj_index, sub_interval_index, obj_interval_index))
            
        ## 9. postprocess some format
        traj_intervals = {k: v.tolist() for k, v in traj_intervals.items()}
        relation_keys = [list(_key) for _key in relation_keys]

        output_dict = {
            "video_hw":(video_anno['height'], video_anno['width']), 
            "relation_merged": relation_merged,
            "relation_keys": relation_keys,
            "visual_features": visual_features,
            "entity_bboxes": entity_bboxes,
            "entity_classes": entity_classes,
            "traj_intervals": traj_intervals,
        }

        return output_dict

    def _train_getitem(self, input_dict, pair_duration=None):
        if len(input_dict.keys()) == 0:
            return {}
        
        data_dict = deepcopy(input_dict)

        ## 1. get the part to load
        relation_merged = data_dict['relation_merged']
        relation_keys = data_dict['relation_keys']

        if pair_duration is not None:
            relation_keys = relation_keys[pair_duration[0]: pair_duration[1]]
            relation_merged = {k: v for k, v in relation_merged.items() if list(k) in relation_keys}

        ## 2. get features
        visual_features = data_dict['visual_features']
        entity_bboxes = data_dict['entity_bboxes']
        traj_intervals = data_dict['traj_intervals']    

        ## bbox clamp
        h_, w_ = data_dict['video_hw']
        for tid in entity_bboxes.keys():
            for interval in range(len(entity_bboxes[tid])):
                entity_bboxes[tid][interval][:, 0] = torch.clamp(entity_bboxes[tid][interval][:, 0], 0)
                entity_bboxes[tid][interval][:, 1] = torch.clamp(entity_bboxes[tid][interval][:, 1], 0)
                entity_bboxes[tid][interval][:, 2] = torch.clamp(entity_bboxes[tid][interval][:, 2], None, w_ - 1)
                entity_bboxes[tid][interval][:, 3] = torch.clamp(entity_bboxes[tid][interval][:, 3], None, h_ - 1)

                assert (torch.all(entity_bboxes[tid][interval][:, 2] > entity_bboxes[tid][interval][:, 0]) and 
                        torch.all(entity_bboxes[tid][interval][:, 3] > entity_bboxes[tid][interval][:, 1]))

        _so_features = []
        _predications = []
        _masks = []

        for relation_key in relation_merged.keys():
            start_offset = random.randint(0, self.feat_stride - 1)
            sub_index, obj_index, sub_interval_index, obj_interval_index = relation_key

            ## 2.1 remove if the relations of one pair is too many 
            if self.cut_max_preds and self.proposal_max_preds < len(relation_merged[relation_key]):
                continue
            
            ## 2.2 so visual features
            sub_interval = traj_intervals[sub_index][sub_interval_index]
            obj_interval = traj_intervals[obj_index][obj_interval_index]
            so_start, so_end = max(sub_interval[0], obj_interval[0]), min(sub_interval[1], obj_interval[1])

            s_start_diff = so_start - sub_interval[0]
            o_start_diff = so_start - obj_interval[0]

            s_feat = visual_features[sub_index][sub_interval_index]
            s_feat = s_feat[s_start_diff: s_start_diff + so_end - so_start]
            s_feat = s_feat[start_offset::self.feat_stride, :]

            o_feat = visual_features[obj_index][obj_interval_index]
            o_feat = o_feat[o_start_diff: o_start_diff + so_end - so_start]
            o_feat = o_feat[start_offset::self.feat_stride, :]

            if s_feat.shape[0] < 2:
                continue

            # 2.3. so bbox features
            sbbox = entity_bboxes[sub_index][sub_interval_index]
            sbbox = sbbox[s_start_diff: s_start_diff + so_end - so_start]
            sbbox = sbbox[start_offset::self.feat_stride, :]

            obbox = entity_bboxes[obj_index][obj_interval_index]
            obbox = obbox[o_start_diff: o_start_diff + so_end - so_start]
            obbox = obbox[start_offset::self.feat_stride, :]

            so_bbox_feat = utils.bbox_to_spatial_features(sbbox, obbox)

            s_bbox_feat = utils.entity_bbox_to_spatial_features(sbbox, h=h_, w=w_)
            o_bbox_feat = utils.entity_bbox_to_spatial_features(obbox, h=h_, w=w_)

            so_feat = torch.cat([s_feat, o_feat, so_bbox_feat, s_bbox_feat, o_bbox_feat], dim=-1).permute(1, 0) # (C, L)

            # 3. pred and mask
            preds = []
            segs = []
            for relation_info in relation_merged[relation_key]:
                predicate_id = relation_info['predicate']

                relation_start = relation_info['begin_fid']
                relation_end = relation_info['end_fid']
                l_, r_ = (relation_start - so_start - start_offset) / self.feat_stride, (relation_end - so_start - start_offset) / self.feat_stride
                l_, r_ = np.ceil(l_), np.ceil(r_)
                
                # if seg is not valid
                if not (l_ < r_):
                    continue
                
                preds.append(predicate_id)
                segs.append(np.array([l_, r_]))
            
            # if no valid preds (too short)
            if len(preds) == 0:
                continue

            ## 4. to tensor
            preds = torch.tensor(preds, dtype=torch.int64)
            segs = torch.tensor(np.array(segs), dtype=torch.int64)
            feats = utils.truncate_feats(so_feat, preds, segs, max_seq_len=self.max_seq_len)
            if feats is None:
                continue
            
            so_feat, preds, segs = feats
            ## 5. seg to mask   
            masks = []
            segs = segs.to(torch.int64)
            for seg in segs:
                mask_ = torch.zeros((self.max_seq_len), dtype=torch.float32)
                assert seg[0] >= 0 and seg[1] <= self.max_seq_len and seg[0] < seg[1]
                mask_[seg[0]: seg[1]] = 1
                masks.append(mask_)

            _so_features.append(so_feat)
            _predications.append(preds)
            _masks.append(torch.stack(masks, dim=0))

        if len(_so_features) == 0:
            return {}

        output_dict = {
            "so_features_list": _so_features,
            "preds_list": _predications,
            "masks_list": _masks,
        } 
        return output_dict

    def _prepare_val(self, video_name):
        """
        proposal_dict: ['MAX_PROPOSAL', 'video_name', 'cat_ids', 'scores', 'bboxes_list', 
                        'traj_durations', 'features_list', 'num_proposals', 'dim_feat', 
                        'video_len', 'video_wh']
        """
        assert self.split == 'validation'
        
        with open(os.path.join(self.info_dir, video_name+'.pkl'), 'rb') as f:
            data_dict = pickle.load(f)
        proposal_dict = data_dict['traj_proposal']

        ## 1. return if info of the video is invalid.
        num_proposals = proposal_dict['num_proposals']
        if num_proposals < 2:
            return {}

        ## 2. get entity infos        
        traj_durations = proposal_dict['traj_durations'].numpy() 
        if self.index_shift:
            traj_durations -= 1
        traj_durations[:, 1] += 1 # left close, right open

        ## 3. get so ids, use meshgrid
        cat_ids = proposal_dict['cat_ids'].numpy()
        so_ids = np.arange(len(cat_ids))
        s_ids, o_ids = np.meshgrid(so_ids, so_ids)
        s_ids, o_ids = s_ids.flatten(), o_ids.flatten()
        s_ids_filtered, o_ids_filtered = s_ids[s_ids != o_ids], o_ids[s_ids != o_ids]

        ## 4. filter invalid meshgird id pairs
        durations_is_valid = []
        for s_id, o_id in zip(s_ids_filtered, o_ids_filtered):
            f_id = max(traj_durations[s_id][0], traj_durations[o_id][0])
            e_id = min(traj_durations[s_id][1], traj_durations[o_id][1])
            if e_id > f_id:
                durations_is_valid.append(True)
            else:
                durations_is_valid.append(False)

        ## 4.1 return if all so pairs are too short
        if True not in durations_is_valid:
            return {}

        ## 5. filter so ids
        durations_is_valid = np.array(durations_is_valid)
        s_ids_filtered, o_ids_filtered = s_ids_filtered[durations_is_valid], o_ids_filtered[durations_is_valid]
        
        ## use numpy first, and change them to tensor later.
        cat_scores = proposal_dict['scores'].numpy()
        bboxes_list = [bboxes.numpy() for bboxes in proposal_dict['bboxes_list']]
        visual_features_list = [features.numpy()[:, :self.dim_visualfeature] for features in proposal_dict['features_list']]

        ## 6. so ids to 
        sids = torch.as_tensor(s_ids_filtered, dtype=torch.int64)
        oids = torch.as_tensor(o_ids_filtered, dtype=torch.int64)
        cat_ids = torch.as_tensor(cat_ids, dtype=torch.int64)
        cat_scores = torch.as_tensor(cat_scores, dtype=torch.float32)
        bboxes_list = [torch.as_tensor(bboxes, dtype=torch.float32) for bboxes in bboxes_list]
        traj_durations = torch.as_tensor(traj_durations, dtype=torch.int64)
        visual_features_list = [torch.as_tensor(feature, dtype=torch.float32) for feature in visual_features_list]

        output_dict = {
            "sids": sids,
            "oids": oids,
            "cat_ids": cat_ids,
            "cat_scores": cat_scores,
            "bboxes_list": bboxes_list,
            "traj_durations": traj_durations,
            "visual_features_list": visual_features_list,
            "video_wh": proposal_dict['video_wh'],
        }
        return output_dict

    def _val_getitem(self, input_dict, viou_threshold=0.9):
        # remove invalid data
        if len(input_dict.keys()) == 0:
            return {}

        data_dict = deepcopy(input_dict)

        ## 1. get features
        sids, oids = data_dict['sids'], data_dict['oids']
        traj_durations = data_dict['traj_durations']
        bboxes_list = data_dict['bboxes_list']
        visual_features_list = data_dict['visual_features_list']

        ## bbox process
        w_, h_ = input_dict['video_wh']
        for tid in range(len(bboxes_list)):
            bboxes_list[tid][:, 0] = torch.clamp(bboxes_list[tid][:, 0], 0)
            bboxes_list[tid][:, 1] = torch.clamp(bboxes_list[tid][:, 1], 0)
            bboxes_list[tid][:, 2] = torch.clamp(bboxes_list[tid][:, 2], None, w_ - 1)
            bboxes_list[tid][:, 3] = torch.clamp(bboxes_list[tid][:, 3], None, h_ - 1)

            assert (torch.all(bboxes_list[tid][:, 2] > bboxes_list[tid][:, 0]) and 
                    torch.all(bboxes_list[tid][:, 3] > bboxes_list[tid][:, 1]))

        ## filter these entity
        num_tids = len(bboxes_list)
        valid_tids = [True] * num_tids

        for base_id in range(num_tids):
            base_bbox = bboxes_list[base_id]

            base_dura = traj_durations[base_id].tolist()
            base_cat_id = data_dict['cat_ids'][base_id]

            for ref_id in range(base_id+1, num_tids):
                if valid_tids[ref_id] == False:
                    continue

                ref_bbox = bboxes_list[ref_id]
                ref_cat_id = data_dict['cat_ids'][ref_id]

                # if catid is different
                if base_cat_id != ref_cat_id:
                    continue
                
                ref_dura = traj_durations[ref_id].tolist()
                if ref_dura[0] >= base_dura[1] or ref_dura[1] <= base_dura[0]:
                    continue

                br_start, br_end = max(base_dura[0], ref_dura[0]), min(base_dura[1], ref_dura[1])

                b_start_diff = br_start - base_dura[0]
                r_start_diff = br_start - ref_dura[0]

                b_bbox = base_bbox[b_start_diff: b_start_diff + br_end - br_start]
                r_bbox = ref_bbox[r_start_diff: r_start_diff + br_end - br_start]

                assert len(b_bbox) == len(r_bbox)

                area_b = (b_bbox[:, 2] - b_bbox[:, 0] + TO_REMOVE) * (
                    b_bbox[:, 3] - b_bbox[:, 1] + TO_REMOVE
                )
                area_r = (r_bbox[:, 2] - r_bbox[:, 0] + TO_REMOVE) * (
                    r_bbox[:, 3] - r_bbox[:, 1] + TO_REMOVE
                )

                lt = torch.max(b_bbox[:, :2], r_bbox[:, :2])
                rb = torch.min(b_bbox[:, 2:], r_bbox[:, 2:])
                wh = (rb - lt + TO_REMOVE).clamp(min=0.0)
                inter_area = (wh[:, 0] * wh[:, 1]).sum()

                viou_br = inter_area / area_r.sum()
                viou_rb = inter_area / area_b.sum()

                if viou_br > viou_threshold and base_dura[0] <= ref_dura[0] and base_dura[1] >= ref_dura[1]:
                    valid_tids[ref_id] = False
                
                elif viou_rb > viou_threshold and ref_dura[0] <= base_dura[0] and ref_dura[1] >= base_dura[1]:
                    valid_tids[base_id] = False
                    break

        valid_tids = torch.tensor(valid_tids, dtype=torch.bool)
        valid_tids = torch.tensor(list(range(num_tids)), dtype=torch.int64)[valid_tids]

        valid_sids = torch.sum((sids[None, :] == valid_tids[:, None]), dim=0) == 1
        valid_oids = torch.sum((oids[None, :] == valid_tids[:, None]), dim=0) == 1
        valid_soids = valid_sids & valid_oids

        sids = sids[valid_soids]
        oids = oids[valid_soids]

        if len(sids) == 0:
            return {}
        
        ## feature
        _so_features_list = []
        _so_offset = []
        
        valid_tids = [True] * len(sids)
        
        for idx, (_sid, _oid) in enumerate(zip(sids, oids)):
            start_offset = random.randint(0, self.feat_stride-1) if self.random_stride else self.stride_offset

            ## 1.1 get so len
            s_start, s_end = traj_durations[_sid][0], traj_durations[_sid][1]
            o_start, o_end = traj_durations[_oid][0], traj_durations[_oid][1]

            so_start, so_end = max(s_start, o_start), min(s_end, o_end)

            bbox_len = so_end - so_start
            s_start_diff = so_start - s_start
            o_start_diff = so_start - o_start

            ## 1.2 so visual features
            s_feat = visual_features_list[_sid][s_start_diff: bbox_len + s_start_diff]
            if s_feat.shape[0] < self.proposal_min_frames:
                valid_tids[idx] = False
                continue
            
            s_feat = s_feat[start_offset::self.feat_stride, :]

            o_feat = visual_features_list[_oid][o_start_diff: bbox_len + o_start_diff]
            o_feat = o_feat[start_offset::self.feat_stride, :]

            if s_feat.shape[0] < 2:
                valid_tids[idx] = False
                continue
            
            ## 1.3 so bbox features
            sbbox = bboxes_list[_sid][s_start_diff: bbox_len + s_start_diff]
            sbbox = sbbox[start_offset::self.feat_stride, :]

            obbox = bboxes_list[_oid][o_start_diff: bbox_len + o_start_diff]
            obbox = obbox[start_offset::self.feat_stride, :]

            so_bbox_feat = utils.bbox_to_spatial_features(sbbox, obbox)

            s_bbox_feat = utils.entity_bbox_to_spatial_features(sbbox, h=h_, w=w_)
            o_bbox_feat = utils.entity_bbox_to_spatial_features(obbox, h=h_, w=w_)

            _so_offset.append(start_offset)
            _so_features_list.append(torch.cat([s_feat, o_feat, so_bbox_feat, s_bbox_feat, o_bbox_feat], dim=-1).permute(1, 0))

        valid_tids = torch.tensor(valid_tids, dtype=torch.bool)
        sids = sids[valid_tids]
        oids = oids[valid_tids]
        
        if len(sids) == 0:
            return {}
        
        _so_offset = torch.tensor(_so_offset, dtype=torch.int64)
        assert len(_so_offset) == len(sids)
        assert len(_so_features_list) == len(sids)
        
        output_dict = {
            "sids": sids,
            "oids": oids,
            "cat_ids": data_dict['cat_ids'],
            "cat_scores": data_dict['cat_scores'],
            "traj_durations": data_dict['traj_durations'],
            "bboxes_list": data_dict['bboxes_list'],
            "so_features_list": _so_features_list,
            "so_offset": _so_offset,
        }
        return output_dict
    

    def __getitem__(self, idx):
        if self.split == 'training':
            policy = self.policy[idx]
            output_dict = dict()

            for policy_info in policy:
                video_name, (pair_start, pair_end) = policy_info[0], policy_info[1]
                ## get data
                with open(os.path.join(self.cache_path, video_name+'.pkl'), 'rb') as f:
                    input_dict = pickle.load(f)
                ## process
                data_dict = self._train_getitem(input_dict, (pair_start, pair_end))

                ## merge
                if len(data_dict.keys()) == 0:
                    continue

                if len(output_dict.keys()) == 0:
                    output_dict = data_dict
                else:
                    assert set(list(output_dict.keys())) == set(list(data_dict.keys()))
                    for k, v in data_dict.items():
                        if isinstance(v, list):
                            output_dict[k] = output_dict[k] + v
                        else:
                            raise TypeError()

        else:
            video_name = self.video_name_list[idx]
            input_dict = self.video_features[video_name]
            output_dict = self._val_getitem(input_dict)

        if len(output_dict.keys()) != 0:
            if self.split == 'validation':
                ## update video name
                output_dict['video_name'] = video_name

            return output_dict
        elif self.split == 'training':
            idx = random.randint(0, len(self.policy) - 1)
            return self.__getitem__(idx)
        else:
            return None
        

    def __len__(self):
        if self.split == 'training':
            return len(self.policy)
        else:
            return len(self.video_name_list)

    @property
    def collator_func(self):
        def collate_fn_train(batch):
            batch_output_dict = defaultdict(list)
            for output_dict in batch:
                for k, v in output_dict.items():
                    if isinstance(v, list):
                        batch_output_dict[k] += v
                    else:
                        raise ValueError()
                            
            return batch_output_dict
        
        def collate_fn_val(batch):
            assert len(batch) == 1
            return batch[0]
        
        if self.split == 'training':
            return collate_fn_train
        else:
            return collate_fn_val
