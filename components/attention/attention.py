"""Attention / RoPEAttention（旋转位置编码多头注意力）。

自实现的多头注意力模块（参考主流 Transformer 架构设计）。
- Attention: 标准多头注意力, 支持对 key/value 降维 (kv_in_dim 可以比 query 维度小)。
- RoPEAttention: 在 Attention 基础上加二维轴向 RoPE, 让注意力感知 token 的相对空间位置。
"""

import math
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from components.attention.rope import apply_rotary_enc, compute_axial_cis


class Attention(nn.Module):
    """标准多头注意力（SDPA 实现）。

    例: 输入 q/k/v 都是 (B, N, 256) → 输出 (B, N, 256)
    """

    def __init__(
        self,
        embedding_dim: int,  # query 的维度，也是输出维度，例 256
        num_heads: int,  # 头数，例 1
        downsample_rate: int = 1,  # 内部维度缩放: internal_dim = embedding_dim / rate
        dropout: float = 0.0,
        kv_in_dim: int = None,  # key/value 的输入维度; None 则等于 embedding_dim
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        # q 从 embedding_dim 投影; k/v 从 kv_in_dim 投影 (可不同, 例跨注意力 k 来自 64 维记忆)
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        # (B, N, C) -> (B, nHead, N, C_per_head)
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        # (B, nHead, N, C_per_head) -> (B, N, C)
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention (PyTorch 官方融合实现, 内含 softmax + 缩放 √d)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class RoPEAttention(Attention):
    """Attention with rotary position encoding.

    在标准注意力前, 先把 RoPE 频率乘进 q 和【部分】k, 使注意力自带相对位置感。
    与正弦位置编码的区别: 不把位置加进 token, 而是作为旋转矩阵作用在 q/k 上,
    好处是任意长度/长度差都能直接算, 不用插值。
    """

    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        # whether to repeat q rope to match k length
        # this is needed for cross-attention to memories
        rope_k_repeat=False,  # 跨注意力(记忆比 query 长)时, 把 q 的频率沿 key 长度重复
        feat_sizes=(64, 64),  # [w, h] for stride 16 feats at 1024 resolution
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # 预计算频率表: 按最大网格 feat_sizes=(64,64) 生成, 例 internal_dim=256 → 每头 256 条频率
        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = (
            freqs_cis.to("cuda") if torch.cuda.is_available() else freqs_cis
        )
        self.rope_k_repeat = rope_k_repeat

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        # num_k_exclude_rope: key 序列里【最后几个】不施加 RoPE 的 token。
        #   在视频跟踪里是"物体指针 token"(object pointers) —— 它们对应具体对象而非空间位置,
        #   所以不该有空间位置编码。
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)  # (B, H, Nq, d)
        k = self._separate_heads(k, self.num_heads)  # (B, H, Nk, d)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        w = h = math.sqrt(q.shape[-2])  # 由 token 数反推网格边长(假设正方形)
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            # token 数不等于预计算的 64×64=4096 时, 按当前实际网格重新算频率
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat  # q 与 k token 数不同 → 必须开了 rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        # 只对前 num_k_rope 个 key 施加 RoPE, 后 num_k_exclude_rope 个(物体指针)保持原样
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
