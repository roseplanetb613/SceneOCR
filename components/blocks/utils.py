"""通用小工具：get_clones / get_activation_fn / get_1d_sine_pe。"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_clones(module, N):
    """深拷贝 N 份同一个模块组成 ModuleList（结构相同、权重独立）。"""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_activation_fn(activation):
    """字符串 → 激活函数。"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """一维正弦位置编码：例 输入 [1,2,3] + dim=64 → (3, 64) 的编码向量。"""
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed
