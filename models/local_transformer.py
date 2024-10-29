from typing import Optional
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .blocks import LayerNorm, AffineDropPath, MaskedMHA, MaskedConv1D, LocalMaskedMHA
from .transformer import _get_clones
from .weight_init import trunc_normal_


class MaskedMHA_QKV(MaskedMHA):
    """
    Multi Head Attention with mask

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """
    def __init__(
        self,
        n_embd,          # dimension of the input embedding
        n_head,          # number of heads in multi-head self-attention
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0   # dropout rate for projection op
    ):
        super().__init__(
            n_embd=n_embd,          
            n_head=n_head,          
            attn_pdrop=attn_pdrop,  
            proj_pdrop=proj_pdrop, 
        )
            
    def forward(self, q, k, v, _qx_mask, _kv_mask, _attn_mask=None):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = q.size()

        # calculate query, key, values for all heads in batch
        # (B, nh * hs, T)
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # move head forward to be the batch dim
        # (B, nh * hs, T) -> (B, nh, T, hs)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to invalid tokens
        if _attn_mask is not None:
            att = att.masked_fill(torch.logical_not(_attn_mask[:, None, :, :]), float('-inf'))
        else:
            att = att.masked_fill(torch.logical_not(_kv_mask[:, :, None, :]), float('-inf'))
        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = att @ (v * _kv_mask[:, :, :, None].to(v.dtype))
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * _qx_mask.to(out.dtype)
        return out, _qx_mask

class MaskedMHCA_QKV(nn.Module):
    """
    Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,          # dimension of the output features
        n_head,          # number of heads in multi-head self-attention
        n_qx_stride=0,
        n_kv_stride=1,   # downsampling stride for key and value
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # conv/pooling operations
        assert (n_qx_stride == 0) or (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_qx_stride == 0) or (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query kernel size = 1, and there weren't any conv among queries
        qx_kernel_size = self.n_qx_stride + 1 if (self.n_qx_stride > 1 or self.n_qx_stride == 0) else 3
        stride, padding = self.n_kv_stride if self.n_kv_stride > 0 else 1, qx_kernel_size // 2
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, qx_kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.query_norm = LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kv_kernel_size = self.n_kv_stride + 1 if (self.n_kv_stride > 1 or self.n_kv_stride == 0) else 3
        stride, padding = self.n_kv_stride if self.n_kv_stride > 0 else 1, kv_kernel_size // 2
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kv_kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kv_kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.value_norm = LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, q, k, v, _qx_mask, _kv_mask, _attn_mask=None):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, _ = q.size()

        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(q, _qx_mask)
        q = self.query_norm(q)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(k, _kv_mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(v, _kv_mask)
        v = self.value_norm(v)

        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # self-attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to invalid tokens
        if _attn_mask is not None:
            att = att.masked_fill(torch.logical_not(_attn_mask[:, None, :, :]), float('-inf'))
        else:
            att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, :]), float('-inf'))
        
        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        return out, qx_mask


class LocalMaskedMHA_QKV(LocalMaskedMHA):
    """
    Local Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    The implementation is fairly tricky, code reference from
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/longformer/modeling_longformer.py
    """

    def __init__(
        self,
        n_embd,          # dimension of the output features
        n_head,          # number of heads in multi-head self-attention
        window_size,     # size of the local attention window
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
        use_rel_pe=False # use relative position encoding
    ):
        super().__init__(
            n_embd=n_embd,
            n_head=n_head,
            window_size=window_size,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            use_rel_pe=use_rel_pe
        )
   
    def forward(self, q, k, v, _qx_mask, _kv_mask, _attn_mask=None):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = q.size()

        # step 1: query, key, value transforms & reshape
        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        
        # (B, nh * hs, T) -> (B, nh, T, hs)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # view as (B * nh, T, hs)
        q = q.view(B * self.n_head, -1, self.n_channels).contiguous()
        k = k.view(B * self.n_head, -1, self.n_channels).contiguous()
        v = v.view(B * self.n_head, -1, self.n_channels).contiguous()

        # step 2: compute local self-attention with rel pe and masking
        q *= self.scale
        # chunked query key attention -> B, T, nh, 2w+1 = window_size
        
        att = self._sliding_chunks_query_key_matmul(
            q, k, self.n_head, self.window_overlap)

        # rel pe
        if self.use_rel_pe:
            att += self.rel_pe
        # kv_mask -> B, T'', 1
        inverse_kv_mask = torch.logical_not(
            _kv_mask[:, :, :, None].view(B, -1, 1))
        # 0 for valid slot, -inf for masked ones
        float_inverse_kv_mask = inverse_kv_mask.type_as(q).masked_fill(
            inverse_kv_mask, -1e4)
        # compute the diagonal mask (for each local window)
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_inverse_kv_mask.new_ones(size=float_inverse_kv_mask.size()),
            float_inverse_kv_mask,
            1,
            self.window_overlap
        )
        att += diagonal_mask

        # ignore input masking for now
        att = nn.functional.softmax(att, dim=-1)
        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        att = att.masked_fill(
            torch.logical_not(_kv_mask.squeeze(1)[:, :, None, None]), 0.0)
        att = self.attn_drop(att)

        # step 3: compute attention value product + output projection
        # chunked attn value product -> B, nh, T, hs
        out = self._sliding_chunks_matmul_attn_probs_value(
            att, v, self.n_head, self.window_overlap)
        # transpose to B, nh, hs, T -> B, nh*hs, T
        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * _qx_mask.to(out.dtype)
        return out, _qx_mask    


class LocalMaskedMHCA_QKV(nn.Module):
    """
    Local Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    The implementation is fairly tricky, code reference from
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/longformer/modeling_longformer.py
    """

    def __init__(
        self,
        n_embd,          # dimension of the output features
        n_head,          # number of heads in multi-head self-attention
        window_size,     # size of the local attention window
        n_qx_stride=0,   # dowsampling stride for query and input
        n_kv_stride=1,   # downsampling stride for key and value
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
        use_rel_pe=False # use relative position encoding
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)
        self.window_size = window_size
        self.window_overlap  = window_size // 2
        # must use an odd window size
        assert self.window_size > 1 and self.n_head >= 1
        self.use_rel_pe = use_rel_pe

        # conv/pooling operations
        assert (n_qx_stride == 0) or (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_qx_stride == 0) or (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        qx_kernel_size = self.n_qx_stride + 1 if (self.n_qx_stride > 1 or self.n_qx_stride == 0) else 3
        stride, padding = self.n_kv_stride if self.n_kv_stride > 0 else 1, qx_kernel_size // 2
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, qx_kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.query_norm = LayerNorm(self.n_embd)
        
        # key, value conv (depthwise)
        kv_kernel_size = self.n_kv_stride + 1 if (self.n_kv_stride > 1 or self.n_kv_stride == 0) else 3
        stride, padding = self.n_kv_stride if self.n_kv_stride > 0 else 1, kv_kernel_size // 2
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kv_kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kv_kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.value_norm = LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # relative position encoding
        if self.use_rel_pe:
            self.rel_pe = nn.Parameter(
                torch.zeros(1, 1, self.n_head, self.window_size))
            trunc_normal_(self.rel_pe, std=(2.0 / self.n_embd)**0.5)

    @staticmethod
    def _chunk(x, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        # x: B x nh, T, hs
        # non-overlapping chunks of size = 2w -> B x nh, T//2w, 2w, hs
        x = x.view(
            x.size(0),
            x.size(1) // (window_overlap * 2),
            window_overlap * 2,
            x.size(2),
        )

        # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2

        # B x nh, #chunks = T//w - 1, 2w, hs
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _pad_and_transpose_last_two_dims(x, padding):
        """pads rows and then flips rows and columns"""
        # padding value is not important because it will be overwritten
        x = nn.functional.pad(x, padding)
        x = x.view(*x.size()[:-2], x.size(-1), x.size(-2))
        return x

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len):
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        # `== 1` converts to bool or uint8
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        ending_mask = ending_mask.expand(ending_input.size())
        # `== 1` converts to bool or uint8
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))

    @staticmethod
    def _pad_and_diagonalize(x):
        """
        shift every row 1 step right, converting columns into diagonals.
        Example::
              chunked_hidden_states: [ 0.4983,  2.6918, -0.0071,  1.0492,
                                       -1.8348,  0.7672,  0.2986,  0.0285,
                                       -0.7584,  0.4206, -0.0405,  0.1599,
                                       2.0514, -1.1600,  0.5372,  0.2629 ]
              window_overlap = num_rows = 4
             (pad & diagonalize) =>
             [ 0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000
               0.0000,  -1.8348,  0.7672,  0.2986,  0.0285, 0.0000,  0.0000
               0.0000,  0.0000, -0.7584,  0.4206, -0.0405,  0.1599, 0.0000
               0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = x.size()
        # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1).
        x = nn.functional.pad(
            x, (0, window_overlap + 1)
        )
        # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        x = x.view(total_num_heads, num_chunks, -1)
        # total_num_heads x num_chunks x window_overlap*window_overlap
        x = x[:, :, :-window_overlap]
        x = x.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        x = x[:, :, :, :-1]
        return x

    def _sliding_chunks_query_key_matmul(
        self, query, key, num_heads, window_overlap
    ):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This implementation splits the input into overlapping chunks of size 2w with an overlap of size w (window_overlap)
        """
        # query / key: B*nh, T, hs
        bnh, seq_len, head_dim = query.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1

        # B * num_heads, head_dim, #chunks=(T//w - 1), 2w
        chunk_query = self._chunk(query, window_overlap)
        chunk_key = self._chunk(key, window_overlap)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        diagonal_chunked_attention_scores = torch.einsum(
            "bcxd,bcyd->bcxy", (chunk_query, chunk_key))

        # convert diagonals into columns
        # B * num_heads, #chunks, 2w, 2w+1
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.
        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1) : -1, window_overlap + 1 :
        ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
            :, 0, : window_overlap - 1, 1 - window_overlap :
        ]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(
        self, attn_probs, value, num_heads, window_overlap
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        bnh, seq_len, head_dim = value.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1
        )

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim)

    def forward(self, q, k, v, _qx_mask, _kv_mask, _attn_mask=None):
        
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, _ = q.size()

        # step 1: depth convolutions
        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(q, _qx_mask)
        q = self.query_norm(q)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(k, _kv_mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(v, _kv_mask)
        v = self.value_norm(v)

        # step 2: query, key, value transforms & reshape
        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        # (B, nh * hs, T) -> (B, nh, T, hs)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # view as (B * nh, T, hs)
        q = q.view(B * self.n_head, -1, self.n_channels).contiguous()
        k = k.view(B * self.n_head, -1, self.n_channels).contiguous()
        v = v.view(B * self.n_head, -1, self.n_channels).contiguous()

        # step 3: compute local self-attention with rel pe and masking
        q *= self.scale
        # chunked query key attention -> B, T, nh, 2w+1 = window_size
        
        att = self._sliding_chunks_query_key_matmul(
            q, k, self.n_head, self.window_overlap)

        # rel pe
        if self.use_rel_pe:
            att += self.rel_pe
        # kv_mask -> B, T'', 1
        inverse_kv_mask = torch.logical_not(
            kv_mask[:, :, :, None].view(B, -1, 1))
        # 0 for valid slot, -inf for masked ones
        float_inverse_kv_mask = inverse_kv_mask.type_as(q).masked_fill(
            inverse_kv_mask, -1e4)
        # compute the diagonal mask (for each local window)
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_inverse_kv_mask.new_ones(size=float_inverse_kv_mask.size()),
            float_inverse_kv_mask,
            1,
            self.window_overlap
        )
        att += diagonal_mask

        # ignore input masking for now
        att = nn.functional.softmax(att, dim=-1)
        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        att = att.masked_fill(
            torch.logical_not(kv_mask.squeeze(1)[:, :, None, None]), 0.0)
        att = self.attn_drop(att)

        # step 4: compute attention value product + output projection
        # chunked attn value product -> B, nh, T, hs
        out = self._sliding_chunks_matmul_attn_probs_value(
            att, v, self.n_head, self.window_overlap)
        # transpose to B, nh, hs, T -> B, nh*hs, T
        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        return out, qx_mask

class MaskedConvTransformerDecoderLayer(nn.Module):
    """
    A simple (post layer norm) Transformer block
    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """
    def __init__(
        self,
        n_embd,                # dimension of the input features
        n_head,                # number of attention heads
        n_hidden=None,         # dimension of the hidden layer in MLP
        act_layer=nn.GELU,     # nonlinear activation used in MLP, default GELU
        attn_pdrop=0.0,        # dropout rate for the attention map
        proj_pdrop=0.0,        # dropout rate for the projection / MLP
        path_pdrop=0.0,        # drop path rate
        n_qx_stride=0,
        n_kv_stride=1,
        with_ffn=True,
        use_local=False,
        win_size=None,
        use_rel_pe=False,
    ):
        super().__init__()
        self.with_ffn = with_ffn

        # layer norm for order (B C T)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)
        
        assert n_qx_stride >= 0 and n_kv_stride >= 0

        if use_local:
            assert win_size is not None
            
            if n_qx_stride == 0:
                self.self_attn = LocalMaskedMHA_QKV(
                    n_embd,
                    n_head,
                    window_size=win_size,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    use_rel_pe=use_rel_pe,
                )
            else:
                self.self_attn = LocalMaskedMHCA_QKV(
                    n_embd,
                    n_head,
                    window_size=win_size,
                    n_qx_stride=n_qx_stride,
                    n_kv_stride=n_kv_stride,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    use_rel_pe=use_rel_pe
                )
            
            if n_kv_stride == 0:
                assert n_qx_stride == 0
                self.multihead_attn = LocalMaskedMHA_QKV(
                    n_embd,
                    n_head,
                    window_size=win_size,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    use_rel_pe=use_rel_pe,
                )
            else:
                self.multihead_attn = LocalMaskedMHCA_QKV(
                    n_embd,
                    n_head,
                    window_size=win_size,
                    n_qx_stride=n_qx_stride,
                    n_kv_stride=n_kv_stride,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    use_rel_pe=use_rel_pe
                )
            
        else:
            if n_qx_stride == 0:
                self.self_attn = MaskedMHA_QKV(
                    n_embd,
                    n_head,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop
                )

            else:
                self.self_attn = MaskedMHCA_QKV(
                    n_embd,
                    n_head,
                    n_qx_stride=n_qx_stride,
                    n_kv_stride=n_qx_stride,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop
                )
                
            if n_kv_stride == 0:
                # if n_kv_stride == 0, n_qx_stride == 0
                assert n_qx_stride == 0

                self.multihead_attn = MaskedMHA_QKV(
                    n_embd,
                    n_head,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop
                )

            else:
                self.multihead_attn = MaskedMHCA_QKV(
                    n_embd,
                    n_head,
                    n_qx_stride=n_qx_stride,
                    n_kv_stride=n_kv_stride,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop
                )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn1 = AffineDropPath(n_embd, drop_prob = path_pdrop)
            self.drop_path_attn2 = AffineDropPath(n_embd, drop_prob = path_pdrop)
        else:
            self.drop_path_attn1 = nn.Identity()
            self.drop_path_attn2 = nn.Identity()

        if with_ffn:
            self.ln3 = LayerNorm(n_embd)

            # two layer mlp
            if n_hidden is None:
                n_hidden = 4 * n_embd  # default

            # ok to use conv1d here with stride=1
            self.mlp = nn.Sequential(
                nn.Conv1d(n_embd, n_hidden, 1),
                act_layer(),
                nn.Dropout(proj_pdrop, inplace=True),
                nn.Conv1d(n_hidden, n_embd, 1),
                nn.Dropout(proj_pdrop, inplace=True),
            )
            
            if path_pdrop > 0.0:
                self.drop_path_mlp = AffineDropPath(n_embd, drop_prob = path_pdrop)
            else:
                self.drop_path_mlp = nn.Identity()

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        cross_first: bool = False,
        attn_mask: Optional[Tensor] = None,
    ):  
        if cross_first:
            tgt2 = self.ln2(tgt)
            tgt2, tgt2_mask = self.multihead_attn(
                q=self.with_pos_embed(tgt2, query_pos),
                k=self.with_pos_embed(memory, pos),
                v=memory, 
                _qx_mask=tgt_mask, 
                _kv_mask=memory_mask,
                _attn_mask=attn_mask,
            )

            tgt2_mask_float = tgt2_mask.to(tgt2.dtype)
            tgt = tgt * tgt2_mask_float + self.drop_path_attn2(tgt2)

            tgt2 = self.ln1(tgt)
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2, tgt2_mask = self.self_attn(
                q, k, v=tgt, _qx_mask=tgt_mask, _kv_mask=tgt_mask
            )

            tgt2_mask_float = tgt2_mask.to(tgt2.dtype)
            tgt = tgt * tgt2_mask_float + self.drop_path_attn1(tgt2)

        else:
            tgt2 = self.ln1(tgt)
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2, tgt2_mask = self.self_attn(
                q, k, v=tgt, _qx_mask=tgt_mask, _kv_mask=tgt_mask
            )

            tgt2_mask_float = tgt2_mask.to(tgt2.dtype)
            tgt = tgt * tgt2_mask_float + self.drop_path_attn1(tgt2)


            tgt2 = self.ln2(tgt)
            tgt2, tgt2_mask = self.multihead_attn(
                q=self.with_pos_embed(tgt2, query_pos),
                k=self.with_pos_embed(memory, pos),
                v=memory, 
                _qx_mask=tgt_mask, 
                _kv_mask=memory_mask,
                _attn_mask=attn_mask,
            )

            tgt2_mask_float = tgt2_mask.to(tgt2.dtype)
            tgt = tgt * tgt2_mask_float + self.drop_path_attn2(tgt2)

        if self.with_ffn:
            tgt2 = self.ln3(tgt)
            tgt = tgt + self.drop_path_mlp(self.mlp(tgt2) * tgt2_mask_float)
            
        return tgt, tgt2_mask
    

class MaskedConvTransformerDecoder(nn.Module):
    def __init__(
        self, 
        n_embd,
        n_head,
        n_hidden,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.1,
        n_qx_stride=0,
        n_kv_stride=1,
        num_layers=4, 
        norm=None, 
        return_intermediate=False,
        use_local=False,
        win_size=None,
        use_rel_pe=False,
    ):
        super().__init__()
        decoder_layer = MaskedConvTransformerDecoderLayer(
            n_embd,
            n_head,
            n_hidden,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            n_qx_stride=n_qx_stride,
            n_kv_stride=n_kv_stride,
            use_local=use_local,
            win_size=win_size,
            use_rel_pe=use_rel_pe,
        )
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                cross_first: bool = False,
        ):
        
        output = tgt
        output_mask = tgt_mask
        intermediate = []
        intermediate_mask = []

        for layer in self.layers:
            output, output_mask = layer(output, memory, tgt_mask=output_mask,
                           memory_mask=memory_mask,
                           pos=pos, query_pos=query_pos, cross_first=cross_first)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_mask.append(output_mask)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_mask)

        return output.unsqueeze(0), output_mask.unsqueeze(0)


class MaskedConvTransformerDecoderOnly(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        n_hidden,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.1,
        n_qx_stride=0,
        n_kv_stride=1,
        num_layers=4, 
        return_intermediate=False,        
        use_local=False,
        win_size=None,
        use_rel_pe=False,
    ):
        super().__init__()

        decoder_norm = LayerNorm(n_embd)
        self.decoder = MaskedConvTransformerDecoder(
            n_embd,
            n_head,
            n_hidden,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            n_qx_stride=n_qx_stride,
            n_kv_stride=n_kv_stride,
            num_layers=num_layers, 
            norm=decoder_norm, 
            return_intermediate=return_intermediate,
            use_local=use_local,
            win_size=win_size,
            use_rel_pe=use_rel_pe,
        )

        self.n_embd = n_embd
        self.n_head = n_head

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, src, mask, query_embed, pos_embed=None, cross_first=False):
        bs_, c_, l_ = src.shape
        
        query_embed = query_embed.unsqueeze(0).repeat(bs_, 1, 1)
        if pos_embed is not None:
            pos_embed = pos_embed.unsqueeze(0).repeat(bs_, 1, 1)
        
        tgt = torch.zeros_like(query_embed)
        tgt_mask = torch.ones((bs_, 1, query_embed.shape[-1]), dtype=torch.bool, device=tgt.device)

        hs, hs_mask = self.decoder(
            tgt, 
            memory=src,
            tgt_mask=tgt_mask, 
            memory_mask=mask, 
            pos = pos_embed,
            query_pos=query_embed,
            cross_first=cross_first,
        )

        return hs, hs_mask

