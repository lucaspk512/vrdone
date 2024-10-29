from collections import defaultdict
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F
from torch import nn

from .backbones import MaskConvTransformerBackbone
from .fpns import FPN1D_Fuse
from .predictor import MaskedTransformerPredictor
from .losses import batch_masked_sigmoid_focal_loss, batch_masked_dice_loss
from .losses import masked_sigmoid_focal_loss, masked_dice_loss

class MaskVRD(nn.Module):
    """
        Transformer based model for single stage relation detection
    """
    def __init__(self, config, device):
        super(MaskVRD, self).__init__()

        ## 1. config
        self.visual_dim = config['visual_dim']
        self.bbox_entity_dim = config['bbox_entity_dim']
        self.bbox_so_dim = config['bbox_so_dim']
        self.embd_dim = config['embd_dim']

        ## other config
        self.max_so_pair = config['max_so_pair']

        ## cost and loss factor
        self.loss_types = config['loss_types']
        self.cost_factor = config["cost_coeff_dict"]
        self.loss_factor = config["loss_coeff_dict"]

        ## loss weight
        # e.g., num_classes = 10 -> 0, 1, ..., 9 as actions, 0 as background
        empty_weight = torch.ones(config["num_classes"] + 1)
        empty_weight[0] = self.loss_factor['eos_coef']
        self.register_buffer("empty_weight", empty_weight)

        # re-distribute params to backbone / neck / head
        self.backbone_arch = tuple(config['backbone_arch'])
        self.scale_factor = config['scale_factor']
        self.fpn_strides = [self.scale_factor**i for i in range(config['fpn_start_level'], self.backbone_arch[-1] + 1)]

        # check the feature pyramid and local attention window size
        self.max_seq_len = config['max_seq_len']
        self.mha_win_size = [config['n_mha_win_size']] * (1 + self.backbone_arch[-1])

        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert self.max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor 

        self.use_abs_pe = config['use_abs_pe']
        self.use_rel_pe = config['use_rel_pe']    


        ## 2.backbone
        self.backbone = MaskConvTransformerBackbone(
            n_visual = self.visual_dim,
            n_bbox_entity = self.bbox_entity_dim,
            n_bbox_so = self.bbox_so_dim,
            n_embd = self.embd_dim,
            n_head = config['n_head'],
            n_embd_ks = config['embd_kernel_size'],
            visual_bbox_ks = config['visual_bbox_fuse_kernel_size'],
            so_fuse_kernel_size = config['so_fuse_kernel_size'],
            n_fuse_head = config['fuse_head'],
            fuse_path_drop = config['fuse_path_drop'],
            fuse_qx_stride = config['fuse_qx_stride'],
            fuse_kv_stride = config['fuse_kv_stride'],
            max_len = self.max_seq_len,
            arch = self.backbone_arch,
            mha_win_size = self.mha_win_size,
            scale_factor = self.scale_factor,
            with_ln = config['embd_with_ln'],
            attn_pdrop = config['dropattn'],
            proj_pdrop = config['dropout'],
            path_pdrop = config['droppath'],
            use_abs_pe = self.use_abs_pe,
            use_rel_pe = self.use_rel_pe,
            use_local = config['use_local'],
        )
    
        if isinstance(self.embd_dim, (list, tuple)):
            self.embd_dim = sum(self.embd_dim)

        ## 3. fpn network: convs
        self.neck = FPN1D_Fuse(
            in_channels = [self.embd_dim] * (self.backbone_arch[-1] + 1),
            out_channel = config['fpn_dim'],
            scale_factor = self.scale_factor,
            start_level = config['fpn_start_level'],
            with_ln = config['fpn_with_ln'],
            norm_first = config['fpn_norm_first'],
        )

        ## 4.decoder
        self.predictor = MaskedTransformerPredictor(**config['predictor'])
        self.deep_supervision = config['predictor']['deep_supervision']

        self.device = device

    @torch.no_grad()
    def _config_eval(self, infer_config):
        """
        add config when evaluation
        """
        assert self.training == False
        self.topk = infer_config['topk']
        self.n_max_pair = infer_config['n_max_pair']
        self.feat_stride = infer_config['feat_stride']
        self.pred_min_frames = infer_config['pred_min_frames']

    def forward(self, input_data):
        if self.training:
            return self.forward_training(input_data)
        else:
            return self.forward_test(input_data)

    def _mask_vrd(self, batched_inputs, batched_masks):
        # forward the network (backbone -> neck -> predictor)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feat, _ = self.neck(feats, masks)

        predictions = self.predictor(feats[-1], fpn_feat, masks[-1], output_mask=masks[0])
        return predictions

    def forward_training(self, input_data):
        ## preprocess, batch the video list into feats (B, C, T) and masks (B, 1, T)
        batched_inputs, batched_masks = self.preprocessing(input_data['so_features_list'])
        predictions = self._mask_vrd(batched_inputs, batched_masks)

        ## match 
        pred_logits, pred_masks = predictions['pred_logits'], predictions['pred_masks']

        gt_preds = input_data['preds_list']
        gt_masks = input_data['masks_list']

        ## bipartite_match
        indices, loss_mask = self.bipartite_match(pred_logits, gt_preds, pred_masks, gt_masks, _mask=predictions['output_mask'])

        ## loss
        loss_dict = self.loss(
            indices, pred_logits, pred_masks, gt_preds, gt_masks, _mask=predictions['output_mask'], loss_mask=loss_mask,
            aux_outputs=(predictions['aux_outputs'] if self.deep_supervision else None),
        )

        total_loss = torch.stack(list(loss_dict.values())).sum()  # scalar tensor
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict

    @torch.no_grad()
    def forward_test(self, input_data):
        bs = len(input_data['sids'])

        ## devide the so pairs with maximum batch
        predictions = defaultdict(list)
        _keys = ['pred_logits', 'pred_masks', 'output_mask']

        for slice_id in range(0, bs, self.max_so_pair):
            slice_input_data = input_data['so_features_list'][slice_id: slice_id + self.max_so_pair]
            batch_inputs, batch_masks, batch_ids = self.preprocessing(slice_input_data)
            
            batched_short_inputs, batched_long_inputs = batch_inputs
            batched_short_masks, batched_long_masks = batch_masks
            short_ids, long_ids = batch_ids

            ## no duplicate or overlap ids
            assert len(set(short_ids + long_ids)) == len(short_ids) + len(long_ids) and len(slice_input_data) == len(short_ids) + len(long_ids)
            slice_len = len(short_ids) + len(long_ids)

            ## merge short and long prediction
            slice_predictions_short, slice_predictions_long = None, None

            if batched_short_inputs is not None:
                slice_predictions_short = self._mask_vrd(batched_short_inputs, batched_short_masks)

            if batched_long_inputs is not None:
                slice_predictions_long = self._mask_vrd(batched_long_inputs, batched_long_masks)

            slice_predictions = [slice_predictions_short, slice_predictions_long]
            ids_to_pred_ids = dict()
            for sidx in range(len(short_ids)):
                ids_to_pred_ids[short_ids[sidx]] = [0, sidx]
            for lidx in range(len(long_ids)):
                ids_to_pred_ids[long_ids[lidx]] = [1, lidx]
            
            for slice_index in range(slice_len):
                _part_type, _part_idx = ids_to_pred_ids[slice_index][0], ids_to_pred_ids[slice_index][1]
                for key in _keys:
                    predictions[key].append(slice_predictions[_part_type][key][_part_idx])

        ## remove pairs without interaction
        pred_logits, pred_masks, output_masks = predictions['pred_logits'], predictions['pred_masks'], predictions['output_mask']
        pred_logits = torch.stack(pred_logits, dim=0)

        pred_scores_list, pred_catids_list, pred_masks_list = [], [], []

        pred_probs = F.softmax(pred_logits, dim=-1)
        pred_scores, pred_catids = torch.topk(pred_probs[..., 1:], k=self.topk, dim=-1)
        for i in range(bs):
            pred_catids_list.append(pred_catids[i].flatten() + 1)
            pred_scores_list.append(pred_scores[i].flatten())
            pred_masks_list.append(pred_masks[i][:, None, :].repeat(1, self.topk, 1).flatten(0, 1))


        ## to results
        triplets = []
        triple_scores = []
        so_trajs = []
        pred_durations = []
        so_tids = []

        for so_idx, (s_id, o_id) in enumerate(zip(input_data['sids'], input_data['oids'])):
            if len(pred_catids_list[so_idx]) == 0:
                continue
            
            ## info
            s_trajs = input_data['bboxes_list'][s_id]
            o_trajs = input_data['bboxes_list'][o_id]
            s_cat_id = input_data['cat_ids'][s_id]
            o_cat_id = input_data['cat_ids'][o_id]
            s_cat_score = input_data['cat_scores'][s_id]
            o_cat_score = input_data['cat_scores'][o_id]

            ## so duration
            s_traj_duration = input_data['traj_durations'][s_id]
            o_traj_duration = input_data['traj_durations'][o_id]
            so_start, so_end = max(s_traj_duration[0], o_traj_duration[0]), min(s_traj_duration[1], o_traj_duration[1])
            s_start_diff = so_start - s_traj_duration[0]
            o_start_diff = so_start - o_traj_duration[0]

            ## raw length
            raw_len = so_end - so_start
            valid_len = torch.sum(output_masks[so_idx])
            stride_offset = input_data['so_offset'][so_idx]
            for uid, (p_cat_score, p_cat_id) in enumerate(zip(pred_scores_list[so_idx], pred_catids_list[so_idx])):
                ## get masks
                mask = pred_masks_list[so_idx][uid].sigmoid() > 0.5
                mask = mask[:valid_len]
                true_indices, _ = torch.sort(torch.nonzero(mask).squeeze(1))
                
                if len(true_indices) == 0:
                    continue
                
                start_index = true_indices[0] * self.feat_stride + stride_offset
                end_index = true_indices[-1] * self.feat_stride + stride_offset + 1

                assert start_index >= 0 and end_index <= raw_len
                if (end_index - start_index) < self.pred_min_frames:
                    continue

                pred_durations.append(torch.stack([so_start + start_index, so_start + end_index], dim=0))
                s_traj_pred = s_trajs[s_start_diff + start_index: s_start_diff + end_index]
                o_traj_pred = o_trajs[o_start_diff + start_index: o_start_diff + end_index]
                assert len(s_traj_pred) == len(o_traj_pred)

                so_trajs.append(torch.stack([s_traj_pred, o_traj_pred], dim=0).tolist())
                triplets.append(torch.stack([s_cat_id, p_cat_id, o_cat_id], dim=0))
                triple_scores.append(torch.stack([s_cat_score, p_cat_score, o_cat_score], dim=0))
                so_tids.append(torch.stack([s_id, o_id], dim=0))

        if len(triplets) == 0:
            return None

        triplets = torch.stack(triplets, dim=0)
        triple_scores = torch.stack(triple_scores, dim=0)
        pred_durations = torch.stack(pred_durations, dim=0)
        so_tids = torch.stack(so_tids, dim=0)

        ## topk so pairs
        triple_scores_avg = torch.mean(triple_scores, dim=-1)
        index = torch.argsort(triple_scores_avg, descending=True)
        index = index[:self.n_max_pair]
        triplets = triplets[index].cpu().tolist()
        triple_scores = triple_scores[index].cpu().tolist()
        triple_scores_avg = triple_scores_avg[index].cpu().tolist()
        pred_durations = pred_durations[index].cpu().tolist()
        so_tids = so_tids[index].cpu().tolist()
        so_trajs = [so_trajs[id] for id in index]
    
        return {
            "triplets": triplets,
            "triple_scores": triple_scores,
            "triple_scores_avg": triple_scores_avg,
            "so_trajs": so_trajs,
            "pred_durations": pred_durations,
            "so_tids": so_tids,
        }

    @torch.no_grad()
    def preprocessing(self, feats_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        if self.training:
            feats_lens = torch.as_tensor([feat.shape[1] for feat in feats_list])
            max_len = feats_lens.max(0).values.item()

            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats_list), feats_list[0].shape[0], max_len]
            batched_inputs = feats_list[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats_list, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
            
            # generate the mask
            batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]
            # push to device
            batched_masks = batched_masks.unsqueeze(1).to(self.device)
            return batched_inputs, batched_masks
        
        else:
            short_feats = [feat for feat in feats_list if feat.shape[1] <= self.max_seq_len]
            long_feats = [feat for feat in feats_list if feat.shape[1] > self.max_seq_len]

            short_ids = [idx for idx in range(len(feats_list)) if feats_list[idx].shape[1] <= self.max_seq_len]
            long_ids = [idx for idx in range(len(feats_list)) if feats_list[idx].shape[1] > self.max_seq_len]

            short_len = torch.as_tensor([feat.shape[1] for feat in short_feats])
            long_len = torch.as_tensor([feat.shape[1] for feat in long_feats])
            
            max_len = self.max_seq_len
            if len(long_feats) > 0:
                for _feat in long_feats:
                    max_len = max(max_len, _feat.shape[1])
            
            stride = self.max_div_factor
            max_len = (max_len + (stride - 1)) // stride * stride

            if len(short_feats) > 0:
                batch_short_shape = [len(short_feats), short_feats[0].shape[0], self.max_seq_len]
                batched_short_inputs = short_feats[0].new_full(batch_short_shape, padding_val)
                for feat, pad_feat in zip(short_feats, batched_short_inputs):
                    pad_feat[..., :feat.shape[-1]].copy_(feat)
                
                # generate the mask
                batched_short_masks = torch.arange(self.max_seq_len)[None, :] < short_len[:, None]

                # push to device
                batched_short_inputs = batched_short_inputs.to(self.device)
                batched_short_masks = batched_short_masks.unsqueeze(1).to(self.device)
            else:
                batched_short_inputs = None
                batched_short_masks = None
            
            if len(long_feats) > 0:
                batch_long_shape = [len(long_feats), long_feats[0].shape[0], max_len]
                batched_long_inputs = long_feats[0].new_full(batch_long_shape, padding_val)
                for feat, pad_feat in zip(long_feats, batched_long_inputs):
                    pad_feat[..., :feat.shape[-1]].copy_(feat)
                
                # generate the mask
                batched_long_masks = torch.arange(max_len)[None, :] < long_len[:, None]

                # push to device
                batched_long_inputs = batched_long_inputs.to(self.device)
                batched_long_masks = batched_long_masks.unsqueeze(1).to(self.device)
            
            else:
                batched_long_inputs = None
                batched_long_masks = None

            return (batched_short_inputs, batched_long_inputs), (batched_short_masks, batched_long_masks), (short_ids, long_ids)
    


    @torch.no_grad()
    def bipartite_match(
        self,
        pred_logits,
        gt_preds,
        pred_masks,
        gt_masks,
        _mask,
    ):  
        bs, num_queries = pred_logits.shape[:2]
        
        out_logits = pred_logits.flatten(0, 1)
        out_mask = pred_masks.flatten(0, 1)
        
        batch_out_mask = _mask.repeat(1, num_queries, 1).flatten(0, 1)
        
        tgt_ids = torch.cat(gt_preds, dim=0)
        tgt_mask = torch.cat(gt_masks, dim=0)
        
        batch_tgt_mask = []
        for i in range(bs):
            batch_tgt_mask += [_mask[i, 0]] * len(gt_masks[i])
        
        batch_tgt_mask = torch.stack(batch_tgt_mask, dim=0)
        assert tgt_mask.shape == batch_tgt_mask.shape
        
        ## class
        out_logits = out_logits[:, :, None].repeat(1, 1, len(tgt_ids))
        tgt_ids = tgt_ids[None, :].repeat(out_logits.shape[0], 1)
        cost_class = F.cross_entropy(out_logits, tgt_ids, reduction='none')

        ## Compute the focal loss between masks
        cost_mask = batch_masked_sigmoid_focal_loss(out_mask, tgt_mask, batch_out_mask=batch_out_mask, batch_tgt_mask=batch_tgt_mask)
        
        ## Compute the dice loss betwen masks
        cost_dice = batch_masked_dice_loss(out_mask, tgt_mask, batch_out_mask=batch_out_mask, batch_tgt_mask=batch_tgt_mask)

        ## Final cost matrix
        _C = (
            self.cost_factor['cost_class'] * cost_class + 
            self.cost_factor['cost_mask'] * cost_mask + 
            self.cost_factor['cost_dice'] * cost_dice
        )    
        
        _C = _C.view(bs, num_queries, -1).cpu()
        sizes = [len(_preds) for _preds in gt_preds]
        
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(_C.split(sizes, -1))]
        loss_mask = torch.cat([torch.stack([_mask[idx].squeeze(0)] * sizes[idx], dim=0) for idx in range(len(sizes))], dim=0)
        
        assert loss_mask.shape[0] == sum(sizes)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], loss_mask
    
    def loss_labels(self, pred_logits, pred_masks, gt_preds, gt_masks, indices, num_masks, loss_mask):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(gt_preds, indices)])

        target_classes = torch.full(
            pred_logits.shape[:2], 0, dtype=torch.int64, device=pred_logits.device
        )

        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_class": self.loss_factor['loss_class'] * loss_ce}
        return losses

    def loss_masks(self, pred_logits, pred_masks, gt_preds, gt_masks, indices, num_masks, loss_mask):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        src_idx = self._get_src_permutation_idx(indices)
        pred_masks = pred_masks[src_idx]
        target_masks = []
        
        for idx, mask in enumerate(gt_masks):
            mask = mask[indices[idx][1]]
            target_masks.append(mask)

        target_masks = torch.cat(target_masks)

        assert pred_masks.shape == target_masks.shape

        losses = {
            "loss_mask": self.loss_factor['loss_mask'] * masked_sigmoid_focal_loss(pred_masks, target_masks, num_masks, loss_mask),
            "loss_dice": self.loss_factor['loss_dice'] * masked_dice_loss(pred_masks, target_masks, num_masks, loss_mask),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, pred_logits, pred_masks, gt_preds, gt_masks, indices, num_masks, loss_mask):
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](pred_logits, pred_masks, gt_preds, gt_masks, indices, num_masks, loss_mask)

    def loss(self, indices, pred_logits, pred_masks, gt_preds, gt_masks, _mask, loss_mask, aux_outputs=None):
        num_masks = sum(len(gt) for gt in gt_preds)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=pred_logits[0].device)
        num_masks = torch.clamp(num_masks, min=1).item()

        losses = {}
        for loss in self.loss_types:
            losses.update(self.get_loss(loss, pred_logits, pred_masks, gt_preds, gt_masks, indices, num_masks, loss_mask))

        if aux_outputs is not None:
            for i, aux_op in enumerate(aux_outputs):
                aux_pred_logits, aux_pred_masks = aux_op['pred_logits'], aux_op['pred_masks']
                
                aux_indices, aux_loss_mask = self.bipartite_match(aux_pred_logits, gt_preds, aux_pred_masks, gt_masks, _mask = _mask)
                for loss in self.loss_types:
                    l_dict = self.get_loss(loss, aux_pred_logits, aux_pred_masks, gt_preds, gt_masks, aux_indices, num_masks, aux_loss_mask)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

