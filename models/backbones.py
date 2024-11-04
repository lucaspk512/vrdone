import torch
from torch import nn
from torch.nn import functional as F

from .blocks import get_sinusoid_encoding, TransformerBlock, MaskedConv1D, LayerNorm, ConvMLP
from .local_transformer import MaskedConvTransformerDecoderLayer

class MaskConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """
    def __init__(
        self,
        n_visual,
        n_bbox_entity,
        n_bbox_so,
        n_embd,
        n_head,
        n_embd_ks,             # conv kernel size of the embedding network
        fuse_ks,
        n_fuse_head, 
        fuse_path_drop,
        fuse_qx_stride,
        fuse_kv_stride,
        max_len,               # max sequence length
        arch = (2, 2, 3),      # (#convs, #stem transformers, #branch transformers)
        mha_win_size = [-1]*4, # size of local window for mha
        scale_factor = 2,      # dowsampling rate for the branch
        with_ln = False,       # if to attach layernorm after conv
        attn_pdrop = 0.0,      # dropout rate for the attention map
        proj_pdrop = 0.0,      # dropout rate for the projection / MLP
        path_pdrop = 0.0,      # droput rate for drop path
        use_abs_pe = False,    # use absolute position embedding
        use_rel_pe = False,    # use relative position embedding
        use_local = True,
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(mha_win_size) == (1 + arch[-1])

        self.n_visual = n_visual
        self.n_bbox_entity = n_bbox_entity
        self.n_bbox_so = n_bbox_so
        
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe

        # embedding network using convs
        self.visual_embd = nn.ModuleList()
        self.visual_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_visual = n_embd if idx > 0 else n_visual
            self.visual_embd.append(
                MaskedConv1D(
                    n_visual, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.visual_embd_norm.append(LayerNorm(n_embd))
            else:
                self.visual_embd_norm.append(nn.Identity())

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        self.bbox_entity_embd = MaskedConv1D(self.n_bbox_entity, n_embd, n_embd_ks, stride=1, padding=n_embd_ks//2)
        self.bbox_entity_norm = LayerNorm(n_embd) if with_ln else nn.Identity()
        
        self.visual_bbox_fuse = ConvMLP(n_embd * 2, n_embd, n_embd, kernel_size=fuse_ks, num_layers=2)

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        self.s_attn = nn.ModuleList()
        self.o_attn = nn.ModuleList()
        
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe
                )
            )

            self.s_attn.append(
                MaskedConvTransformerDecoderLayer(
                    n_embd=n_embd,
                    n_head=n_fuse_head,
                    path_pdrop=fuse_path_drop,
                    n_qx_stride=fuse_qx_stride,
                    n_kv_stride=fuse_kv_stride,
                    with_ffn=False,
                    use_local=use_local,
                    win_size=self.mha_win_size[0] if use_local else None,
                )
            )
            
            self.o_attn.append(
                MaskedConvTransformerDecoderLayer(
                    n_embd=n_embd,
                    n_head=n_fuse_head,
                    path_pdrop=fuse_path_drop,
                    n_qx_stride=fuse_qx_stride,
                    n_kv_stride=fuse_kv_stride,
                    with_ffn=False,
                    use_local=use_local,
                    win_size=self.mha_win_size[0] if use_local else None,
                )
            )
        
        self.s_fuse_norm = LayerNorm(n_embd)
        self.o_fuse_norm = LayerNorm(n_embd)

        self.so_fuse = ConvMLP(n_embd * 2, n_embd, n_embd, kernel_size=fuse_ks, num_layers=2)
        self.bbox_so_embd = MaskedConv1D(self.n_bbox_so, n_embd, n_embd_ks, stride=1, padding=n_embd_ks//2)
        self.so_visual_bbox_fuse = ConvMLP(n_embd * 2, n_embd, n_embd, kernel_size=fuse_ks, num_layers=2)

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe
                )
            )

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)

        assert x.shape[1] == (2 * self.n_visual + self.n_bbox_so + 2 * self.n_bbox_entity)
        
        ## visual
        s_feat, o_feat = x[:, :(self.n_visual)], x[:, (self.n_visual): (2 * self.n_visual)]
        ## bbox relative
        bbox_so_feat = x[:, (2 * self.n_visual): (2 * self.n_visual + self.n_bbox_so)]
        ## bbox entity
        bbox_entity_feat = x[:, (2 * self.n_visual + self.n_bbox_so):]
        s_bbox_feat, o_bbox_feat = bbox_entity_feat[:, :self.n_bbox_entity], bbox_entity_feat[:, self.n_bbox_entity:]
        
        mask_float = mask.to(s_feat.dtype)
        B, _, T = s_feat.size()

        # embedding network
        for idx in range(len(self.visual_embd)):
            s_feat, _ = self.visual_embd[idx](s_feat, mask)
            s_feat = self.relu(self.visual_embd_norm[idx](s_feat))

            o_feat, _ = self.visual_embd[idx](o_feat, mask)
            o_feat = self.relu(self.visual_embd_norm[idx](o_feat))

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            s_feat = s_feat + pe[:, :, :T] * mask.to(s_feat.dtype)
            o_feat = o_feat + pe[:, :, :T] * mask.to(o_feat.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            s_feat = s_feat + pe[:, :, :T] * mask.to(s_feat.dtype)
            o_feat = o_feat + pe[:, :, :T] * mask.to(o_feat.dtype)

        ## bbox feat
        s_bbox_feat, _ = self.bbox_entity_embd(s_bbox_feat, mask)
        s_bbox_feat = self.relu(self.bbox_entity_norm(s_bbox_feat))
        o_bbox_feat, _ = self.bbox_entity_embd(o_bbox_feat, mask)
        o_bbox_feat = self.relu(self.bbox_entity_norm(o_bbox_feat))
        
        ## so entity bbox
        s_feat = self.visual_bbox_fuse(torch.cat([s_feat, s_bbox_feat], dim=1))
        o_feat = self.visual_bbox_fuse(torch.cat([o_feat, o_bbox_feat], dim=1))
        
        s_feat = s_feat * mask_float
        o_feat = o_feat * mask_float

        # stem transformer
        for idx in range(len(self.stem)):
            s_feat, _ = self.stem[idx](s_feat, mask)
            o_feat, _ = self.stem[idx](o_feat, mask)

            # mutual attn
            s_feat_mutual = self.s_attn[idx](s_feat, o_feat, mask, mask)[0]
            o_feat_mutual = self.o_attn[idx](o_feat, s_feat, mask, mask)[0]        

            s_feat = s_feat + s_feat_mutual
            o_feat = o_feat + o_feat_mutual
            
        
        s_feat = self.s_fuse_norm(s_feat)
        o_feat = self.o_fuse_norm(o_feat)

        # so fuse
        so_feat = self.so_fuse(torch.cat([s_feat, o_feat], dim=1))
        so_feat = so_feat * mask_float

        ## so bbox
        bbox_so_feat, _ = self.bbox_so_embd(bbox_so_feat, mask)
        
        so_feat = torch.cat([so_feat, bbox_so_feat], dim=1)
        so_embedding = self.so_visual_bbox_fuse(so_feat)
        so_embedding = so_embedding * mask_float

        # prep for outputs
        out_feats = (so_embedding, )
        out_masks = (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            so_embedding, mask = self.branch[idx](so_embedding, mask)
            out_feats += (so_embedding, )
            out_masks += (mask, )

        return out_feats, out_masks

class MaskConvTransformerBackboneWithCLIP(MaskConvTransformerBackbone):
    def __init__(
        self,
        n_visual,
        n_clip,
        n_bbox_entity,
        n_bbox_so,
        n_embd,
        n_head,
        n_embd_ks,             # conv kernel size of the embedding network
        fuse_ks,
        n_fuse_head, 
        fuse_path_drop,
        fuse_qx_stride,
        fuse_kv_stride,
        max_len,               # max sequence length
        arch = (2, 2, 3),      # (#convs, #stem transformers, #branch transformers)
        mha_win_size = [-1]*4, # size of local window for mha
        scale_factor = 2,      # dowsampling rate for the branch
        with_ln = False,       # if to attach layernorm after conv
        attn_pdrop = 0.0,      # dropout rate for the attention map
        proj_pdrop = 0.0,      # dropout rate for the projection / MLP
        path_pdrop = 0.0,      # droput rate for drop path
        use_abs_pe = False,    # use absolute position embedding
        use_rel_pe = False,    # use relative position embedding
        use_local = True,
    ):
        super().__init__(
            n_visual=n_visual, 
            n_bbox_entity=n_bbox_entity, 
            n_bbox_so=n_bbox_so, 
            n_embd=n_embd,
            n_head=n_head, 
            n_embd_ks=n_embd_ks,
            fuse_ks=fuse_ks, 
            n_fuse_head=n_fuse_head, 
            fuse_path_drop=fuse_path_drop,
            fuse_qx_stride=fuse_qx_stride, 
            fuse_kv_stride=fuse_kv_stride,
            max_len=max_len, 
            arch=arch, 
            mha_win_size=mha_win_size,
            scale_factor=scale_factor, 
            with_ln=with_ln,
            attn_pdrop=attn_pdrop, 
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            use_abs_pe=use_abs_pe,
            use_rel_pe=use_rel_pe,
            use_local=use_local,
        )

        # clip embedding network using convs
        self.n_clip = n_clip
        self.clip_embd = nn.ModuleList()
        self.clip_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_clip = n_embd if idx > 0 else n_clip
            self.clip_embd.append(
                MaskedConv1D(
                    n_clip, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.clip_embd_norm.append(LayerNorm(n_embd))
            else:
                self.clip_embd_norm.append(nn.Identity())
        
        self.visual_clip_fuse = ConvMLP(n_embd * 2, n_embd, n_embd, kernel_size=fuse_ks, num_layers=2)
        self.apply(self.__init_weights__)


    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)

        assert x.shape[1] == (2 * self.n_visual + 2 * self.n_clip + self.n_bbox_so + 2 * self.n_bbox_entity)
        
        feat_part = x[:, :(2 * self.n_visual)]
        clip_part = x[:, (2 * self.n_visual): (2 * self.n_visual + 2 * self.n_clip)]
        bbox_so_part = x[:, (2 * self.n_visual + 2 * self.n_clip): (2 * self.n_visual + 2 * self.n_clip + self.n_bbox_so)]
        bbox_entity_part = x[:, (2 * self.n_visual + 2 * self.n_clip + self.n_bbox_so):]
        
        ## visual
        s_feat, o_feat = feat_part[:, :self.n_visual], feat_part[:, self.n_visual:]
        ## clip
        s_clip, o_clip = clip_part[:, :self.n_clip], clip_part[:, self.n_clip:]
        ## bbox relative
        bbox_so_feat = bbox_so_part
        ## bbox entity
        s_bbox_feat, o_bbox_feat = bbox_entity_part[:, :self.n_bbox_entity], bbox_entity_part[:, self.n_bbox_entity:]
        
        mask_float = mask.to(s_feat.dtype)
        B, _, T = s_feat.size()

        # embedding network
        for idx in range(len(self.visual_embd)):
            s_feat, _ = self.visual_embd[idx](s_feat, mask)
            s_feat = self.relu(self.visual_embd_norm[idx](s_feat))

            o_feat, _ = self.visual_embd[idx](o_feat, mask)
            o_feat = self.relu(self.visual_embd_norm[idx](o_feat))

        for idx in range(len(self.clip_embd)):
            s_clip, _ = self.clip_embd[idx](s_clip, mask)
            s_clip = self.relu(self.clip_embd_norm[idx](s_clip))

            o_clip, _ = self.clip_embd[idx](o_clip, mask)
            o_clip = self.relu(self.clip_embd_norm[idx](o_clip))

        s_feat = self.visual_clip_fuse(torch.cat([s_feat, s_clip], dim=1))
        o_feat = self.visual_clip_fuse(torch.cat([o_feat, o_clip], dim=1))

        s_feat = s_feat * mask_float
        o_feat = o_feat * mask_float
        
        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            s_feat = s_feat + pe[:, :, :T] * mask.to(s_feat.dtype)
            o_feat = o_feat + pe[:, :, :T] * mask.to(o_feat.dtype)
            
        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            s_feat = s_feat + pe[:, :, :T] * mask.to(s_feat.dtype)
            o_feat = o_feat + pe[:, :, :T] * mask.to(o_feat.dtype)

        ## bbox feat
        s_bbox_feat, _ = self.bbox_entity_embd(s_bbox_feat, mask)
        s_bbox_feat = self.relu(self.bbox_entity_norm(s_bbox_feat))
        o_bbox_feat, _ = self.bbox_entity_embd(o_bbox_feat, mask)
        o_bbox_feat = self.relu(self.bbox_entity_norm(o_bbox_feat))
        
         ## so entity bbox
        s_feat = self.visual_bbox_fuse(torch.cat([s_feat, s_bbox_feat], dim=1))
        o_feat = self.visual_bbox_fuse(torch.cat([o_feat, o_bbox_feat], dim=1))
        
        s_feat = s_feat * mask_float
        o_feat = o_feat * mask_float

        # stem transformer
        for idx in range(len(self.stem)):
            s_feat, _ = self.stem[idx](s_feat, mask)
            o_feat, _ = self.stem[idx](o_feat, mask)

            # mutual attn
            s_feat_mutual = self.s_attn[idx](s_feat, o_feat, mask, mask)[0]
            o_feat_mutual = self.o_attn[idx](o_feat, s_feat, mask, mask)[0]        

            s_feat = s_feat + s_feat_mutual
            o_feat = o_feat + o_feat_mutual
            
        
        s_feat = self.s_fuse_norm(s_feat)
        o_feat = self.o_fuse_norm(o_feat)

        # so fuse
        so_feat = self.so_fuse(torch.cat([s_feat, o_feat], dim=1))
        so_feat = so_feat * mask_float

        ## so bbox
        bbox_so_feat, _ = self.bbox_so_embd(bbox_so_feat, mask)
        
        so_feat = torch.cat([so_feat, bbox_so_feat], dim=1)
        so_embedding = self.so_visual_bbox_fuse(so_feat)
        so_embedding = so_embedding * mask_float

        # prep for outputs
        out_feats = (so_embedding, )
        out_masks = (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            so_embedding, mask = self.branch[idx](so_embedding, mask)
            out_feats += (so_embedding, )
            out_masks += (mask, )

        return out_feats, out_masks

