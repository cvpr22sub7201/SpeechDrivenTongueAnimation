# -*- coding: utf-8 -*-
"""
@author: Salvador Medina
"""

from __future__ import annotations
from typing import Callable

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange

__all__ = [
    'TongueFormer'
]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Adds the position encoding to the input tensor

        Args:
            x (Tensor): Torch tensor [Seq x Batch x Embedding_Dim]

        Returns:
            Tensor: Position encoded tensor [Seq x Batch x Embedding_Dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SpeechAnimationTransformer(nn.Module):
    def __init__(self, input_sz, output_sz, d_model,
                 num_heads, num_encoder_layers, num_decoder_layers, embedding_sz,
                 seq_len, pos_dropout, dropout=0.5):
        super(SpeechAnimationTransformer, self).__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.src_fc = nn.Linear(input_sz, d_model)
        self.tgt_fc = nn.Linear(output_sz, d_model)
        self.pos_enc = PositionalEncoding(d_model, seq_len, pos_dropout)

        self.transformer = nn.Transformer(d_model=d_model, nhead=num_heads,
                                          num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=embedding_sz, dropout=dropout)

        self.output_fc = nn.Linear(d_model, output_sz)

        self.params = dict(input_sz=input_sz,
                           output_sz=output_sz,
                           d_model=d_model,
                           num_heads=num_heads,
                           num_encoder_layers=num_encoder_layers,
                           num_decoder_layers=num_decoder_layers,
                           embedding_sz=embedding_sz,
                           seq_len=seq_len,
                           pos_dropout=pos_dropout,
                           dropout=dropout)

    def generate_square_subsequent_mask(self, size):
        """Generates a mask for no peek into the future

        Args:
            size (int): Size of the squared filter mask

        Returns:
            Tensor: Mask with -inf in upper triangle and zero in diag/lower triangle
        """
        mask = rearrange(torch.triu(torch.ones(size, size)) == 1, 'h w -> w h')
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask, mem_key_padding_mask, tgt_mask):
        # Rearrange from batch x seq_len x src/tgt_dim to seq_len x batch x src/tgt_dim
        src = rearrange(src, 'b s d -> s b d')
        tgt = rearrange(tgt, 'b s d -> s b d')

        src_embedding = self.src_fc(src)
        tgt_embedding = self.tgt_fc(tgt)

        src = self.pos_enc(src_embedding * math.sqrt(self.d_model))
        tgt = self.pos_enc(tgt_embedding * math.sqrt(self.d_model))

        output = self.transformer(src, tgt,
                                  tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=mem_key_padding_mask)

        # Rearrange to batch first
        output = rearrange(output, 's b d -> b s d')

        return self.output_fc(output)

    def save_model(self, save_path):
        torch.save({'model_params': self.params,
                    'model_state_dict': self.state_dict()
                    }, save_path)

    @staticmethod
    def load_model(checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location=str(device))
        model = SpeechAnimationTransformer(**checkpoint['model_params'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model


class MLP(nn.Module):
    """Multi Layer Perceptron class"""

    def __init__(self, in_feats: int, hidden_feats: int = None, out_feats: int = None,
                 act_layer: Callable[[torch.Tensor], torch.Tensor] = nn.GELU,
                 drop_rate: float = 0.0):
        """Initialize MLP

        Args:
            in_feats (int): input number of features
            hidden_feats (int, optional): hidden dimension. Defaults to None.
            out_feats (int, optional): output dimension. Defaults to None.
            act_layer (Callable[[torch.Tensor], torch.Tensor], optional): activation function.
                                                                          Defaults to nn.GELU.
            drop_rate (float, optional): dropout. Defaults to 0.0.
        """

        super().__init__()
        hidden_feats = hidden_feats or in_feats
        out_feats = out_feats or in_feats
        self.fc1 = nn.Linear(in_feats, hidden_feats)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_feats, out_feats)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward"""

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Attention class"""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 qk_scale: float = None, attn_drop: float = 0.0, proj_drop: float = 0.0):
        """Initialize module

        Args:
            dim (int): input dimension
            num_heads (int, optional): number of heads. Defaults to 8.
            qkv_bias (bool, optional): Apply bias. Defaults to False.
            qk_scale (float, optional): scale factor to query-key. Defaults to None.
            attn_drop (float, optional): dropout for attention. Defaults to 0.0.
            proj_drop (float, optional): dropout. Defaults to 0.0.
        """

        super().__init__()
        self._num_heads = num_heads
        head_dim = dim // num_heads

        self._scale = head_dim ** -0.5
        self._qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self._attn_drop = nn.Dropout(attn_drop)
        self._proj = nn.Linear(dim, dim)
        self._proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward"""

        B, N, C = x.shape
        qkv_out = self._qkv(x).view(B, N, 3, self._num_heads, C //
                                    self._num_heads)

        # TODO: triggers error with multi-GPU execution due to
        # data on multiple GPUs (batch re-organization)
        qkv = qkv_out.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self._scale
        attn = attn.softmax(dim=-1)
        attn = self._attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self._proj(x)
        x = self._proj_drop(x)
        return x


class MultiHeadedAttentionBlock(nn.Module):
    """Multi-headed attention block"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, qk_scale: float = None, drop: float = 0.0,
                 attn_drop: float = 0.0, dropout_ratio=0.0,
                 act_layer: Callable[[torch.Tensor], torch.Tensor] = nn.GELU,
                 norm_layer: Callable[[torch.Tensor], torch.Tensor] = nn.LayerNorm):
        """Initialize class

        Args:
            dim (int): dimension
            num_heads (int): number of heads
            mlp_ratio (float, optional): How much it changes the input. Defaults to 4..
            qkv_bias (bool, optional): Apply bias. Defaults to False.
            qk_scale (float, optional): scale factor. Defaults to None.
            drop (float, optional): dropout for MLP. Defaults to 0.0.
            attn_drop (float, optional): dropout on attention layer. Defaults to 0.0.
            dropout_ratio (float, optional): drop-out for positional embedding.
                                             Defaults to 0.0.
            act_layer (Callable[[torch.Tensor], torch.Tensor], optional): activation layer.
                                                                          Defaults to nn.GELU.
            norm_layer (Callable[[torch.Tensor], torch.Tensor], optional): normalization layer.
                                                                           Defaults to nn.LayerNorm.
        """
        super().__init__()

        self._attn_norm = norm_layer(dim)
        self._mlp_norm = norm_layer(dim)
        self._attn = Attention(dim, num_heads=num_heads,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               attn_drop=attn_drop,
                               proj_drop=drop)

        if dropout_ratio > 0.0:
            self._drop_path = nn.Dropout(p=dropout_ratio, inplace=True)
        else:
            self._drop_path = nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self._mlp = MLP(in_feats=dim, hidden_feats=mlp_hidden_dim,
                        act_layer=act_layer, drop_rate=drop)

    def forward(self, x) -> torch.Tensor:
        """Forward"""

        x = x + self._drop_path(self._attn(self._attn_norm(x)))
        x = x + self._drop_path(self._mlp(self._mlp_norm(x)))
        return x


class TongueFormer(nn.Module):
    def __init__(self, num_frames=50, in_feat_dim=1024, out_idx=25, out_full=False, num_joints=10,
                 num_layers=4, num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=None, mask=None, no_peek=False, deeper_fc=False):
        """
        Args:
            num_frames (int, tuple): input frame number
            in_feat_dim (int):
            out_idx (int):
            out_full (bool):
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            num_layers (int): num_layers of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for QKV if True
            qk_scale (float): override default QK scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            mask (str): select the mask for the multiheaded attention (no_peek, diagonal, bandwidth_1, bandwidth_2)
            no_peek (bool): TODO: legacy, need to remove after siggraph
        """
        super().__init__()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_frames = num_frames
        self.out_dim = num_joints * 3
        self.out_idx = out_idx
        self.out_full = out_full
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, in_feat_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.mask = mask
        if no_peek:
            self.mask = 'no_peek'

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                num_layers)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            MultiHeadedAttentionBlock(
                dim=in_feat_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, dropout_ratio=dpr[i], norm_layer=norm_layer)
            for i in range(num_layers)])

        self.norm = norm_layer(in_feat_dim)

        if not self.out_full:
            self.weighted_mean = torch.nn.Conv1d(in_channels=num_frames,
                                                 out_channels=1,
                                                 kernel_size=1)

        # adding additional hidden layer with activation if selected
        if deeper_fc:
            self.head = nn.Sequential(
                nn.LayerNorm(in_feat_dim),
                nn.Linear(in_feat_dim, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, self.out_dim)
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(in_feat_dim),
                nn.Linear(in_feat_dim, self.out_dim)
            )

        self.forward_feats = self.forward_feats_full if self.out_full else self.forward_feats_mean

        self.params = dict(num_frames=num_frames,
                           in_feat_dim=in_feat_dim,
                           out_idx=out_idx,
                           out_full=out_full,
                           num_joints=num_joints,
                           num_layers=num_layers,
                           num_heads=num_heads,
                           mlp_ratio=mlp_ratio,
                           qkv_bias=qkv_bias,
                           qk_scale=qk_scale,
                           drop_rate=drop_rate,
                           attn_drop_rate=attn_drop_rate,
                           drop_path_rate=drop_path_rate,
                           norm_layer=norm_layer,
                           mask=mask,
                           no_peek=no_peek,
                           deeper_fc=deeper_fc)

    def forward_feats_mean(self, x):
        b = x.shape[0]
        x += self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.weighted_mean(x)
        x = x.view(b, 1, -1)
        return x

    def forward_feats_full(self, x):
        b = x.shape[0]
        x += self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        """ Transforms from a sequence of audio feature represenations to a tongue, lips and jaw landmark pose
            The pose

        Args:
            x (Tensor): Batch of sequence of audio features [B x F x D]
                        B: batch, F: num frames, D: audio feat dim
        Output:
            (Tensor): Output tensor with target pose of tongue, lips and jaw, [B x O]
                        B: batch, O: output dim
        """
        x = self.forward_feats(x)
        x = self.head(x).squeeze(1)

        return x


class DualTongueformer(nn.Module):
    def __init__(self, num_frames=50, in_feat_dim=1024, out_idx=25, out_full=False, num_joints=10,
                 num_layers=4, num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=None, mask=None, no_peek=False, deeper_fc=False):
        self._encoder = TongueFormer(num_frames, in_feat_dim, out_idx, out_full, num_joints,
                                num_layers, num_heads, mlp_ratio, qkv_bias, qk_scale,
                                drop_rate, attn_drop_rate, drop_path_rate,
                                norm_layer, mask, no_peek, deeper_fc)
        self._decoder = TongueFormer(num_frames, in_feat_dim, out_idx, out_full, num_joints,
                                num_layers, num_heads, mlp_ratio, qkv_bias, qk_scale,
                                drop_rate, attn_drop_rate, drop_path_rate,
                                norm_layer, mask, no_peek, deeper_fc)

        def forward(self, x):
            feats = self._encoder.forward_feats_full(x)
            x = self._decoder.forward_feats(x)


if __name__ == '__main__':
    from torch import nn, optim

    tongueformer = TongueFormer(num_frames=50, in_feat_dim=1024, num_joints=10, num_layers=4,
                                num_heads=8, mlp_ratio=2, qkv_bias=True, qk_scale=None,
                                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None,
                                no_peek=True)
    model_optimizer = optim.Adam(tongueformer.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    dummy_x = torch.Tensor(32, 50, 1024)
    dummy_y = tongueformer(dummy_x)

    loss = criterion(dummy_y, torch.zeros(dummy_y.shape))
    model_optimizer.zero_grad()
    loss.backward()
    model_optimizer.step()

    print(dummy_y.shape)
