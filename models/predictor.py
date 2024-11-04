# ====================================================
# This code is modified by Xinjie Jiang (jiangxinjie512@gmail.com)
# ====================================================

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import math
import torch
from torch import nn
from torch.nn import functional as F

from .blocks import ConvMLP, LayerNorm
from .local_transformer import MaskedConvTransformerDecoderOnly

class MaskedTransformerPredictor(nn.Module):
    def __init__(
        self,
        n_input, 
        n_embd,
        n_head,
        n_hidden,
        num_queries,
        num_classes,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.1,
        cls_prior_prob=0.01,
        n_qx_stride=0,
        n_kv_stride=1,
        num_layers=4, 
        deep_supervision=False,
        enforce_input_project=False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dropout: dropout in Transformer
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            deep_supervision: whether to add supervision to every decoder layers
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()
        self.transformer = MaskedConvTransformerDecoderOnly(
            n_embd,
            n_head,
            n_hidden,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            n_qx_stride=n_qx_stride,
            n_kv_stride=n_kv_stride,
            num_layers=num_layers, 
            return_intermediate=deep_supervision,
        )

        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, n_embd)
        self.input_norm = LayerNorm(n_input)

        self.input_proj = None
        if n_input != n_embd or enforce_input_project:
            self.input_proj = nn.Conv1d(n_input, n_embd, kernel_size=1)
            nn.init.constant_(self.input_proj.bias, 0.)

        self.aux_loss = deep_supervision

        ## output FFNs
        self.class_embed = nn.Conv1d(n_embd, num_classes + 1, 1) # with background
        bias_value = -(math.log((1 - cls_prior_prob) / cls_prior_prob))
        nn.init.constant_(self.class_embed.bias, bias_value)

        self.mask_embed = ConvMLP(n_embd, n_embd, n_embd, 3)

    def forward(self, x, mask_features, mask, output_mask, non_attn_const=(-10)):
        src = self.input_norm(x)
        if self.input_proj is not None:
            src = self.input_proj(src)
            src_mask_float = mask.to(src.dtype)
            src = src * src_mask_float

        hs, _ = self.transformer(src, mask, query_embed=self.query_embed.weight.permute(1, 0))

        n_layer, bs, n_channel, n_query = hs.shape
        hs_embed = hs.view(n_layer*bs, n_channel, n_query)
                
        outputs_class = self.class_embed(hs_embed).view(n_layer, bs, -1, n_query).permute(0, 1, 3, 2)
        out = {"pred_logits": outputs_class[-1]}

        if self.aux_loss:
            # [l, bs, queries, embed]
            mask_embed = self.mask_embed(hs_embed).view(n_layer, bs, -1, n_query).permute(0, 1, 3, 2)
            outputs_seg_masks = torch.einsum("lbqc,bcm->lbqm", mask_embed, mask_features)
            outputs_seg_masks = outputs_seg_masks.masked_fill(torch.logical_not(output_mask[None, ...]), float(non_attn_const))
            out["pred_masks"] = outputs_seg_masks[-1]
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_seg_masks)
        else:
            # [bs, queries, embed]
            mask_embed = self.mask_embed(hs[-1]).permute(0, 2, 1)
            outputs_seg_masks = torch.einsum("bqc,bcm->bqm", mask_embed, mask_features)
            outputs_seg_masks = outputs_seg_masks.masked_fill(torch.logical_not(output_mask), float(non_attn_const))
            out["pred_masks"] = outputs_seg_masks
        
        out['output_mask'] = output_mask
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]
