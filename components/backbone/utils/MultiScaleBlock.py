import logging
from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
# ---- 自建参数 ----
from components.backbone.utils import window_partition, window_unpartition
from components.backbone.utils import (
    Mlp,
    PatchEmbed,
)


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # 例: x (B, H, W, C) = (1, 28, 28, 384)
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    # (1, 384, 28, 28)
    x = pool(x)
    # MaxPool2d(2,2): (1, 384, 14, 14) —— 空间分辨率减半
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    # (1, 14, 14, 384) —— 回到 BHWC
    # 均值化
    if norm:
        x = norm(x)

    return x

class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        """
            B, H, W, C
        """
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.pool = q_pool

        # ---- 初始化权重参数张量 ----
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # 进入这里的 x 是【单个窗口】的批: B = 窗口数, H=W = 窗口边长
        # 例 blk3 (28×28, 窗口4): x (49, 4, 4, 192)   ← 49 个 4×4 窗口
        # 例 blk2 (56×56, 窗口8): x (49, 8, 8, 96)    ← 49 个 8×8 窗口
        # qkv (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # 例 blk3: (49, 16, 3, 2, 96)   → -1 = 每个头分到的维度 192/2=96
        q, k, v = torch.unbind(qkv, 2)  # 各取一维: q,k,v 都是 (B, H*W, nHead, head_dim)
        if self.pool:
            # ======== Q-pooling (只在 stage 切换块发生) ========
            # 只把 Query 用 MaxPool 下采样, Key/Value 保持全分辨率。
            # 于是 query 位置数 < key/value 位置数: 每个位置用少量 query 去"看"整窗的更多特征,
            # 同时把空间分辨率降一半 —— 下采样和注意力一步完成。
            # 例 blk2: q (49, 64, 2, 96) -> reshape (49, 8, 8, 192) -> MaxPool -> (49, 4, 4, 192)
            #          之后 query 只有 4×4=16 个, 而 key/value 仍是 8×8=64 个
            q = do_pool(q.reshape(B, H, W, -1), self.pool)
            H, W = q.shape[1:3]           # 更新成下采样后的窗口边长: 例 (4, 4)
            q = q.reshape(B, H * W, self.num_heads, -1)   # 例 (49, 16, 2, 96)

        x = F.scaled_dot_product_attention(
        q.transpose(1, 2),   # (B, nHead, nQuery, head_dim)   例 (49, 2, 16, 96)
        k.transpose(1, 2),   # (B, nHead, nKey,   head_dim)   例 (49, 2, 64, 96)
        v.transpose(1, 2),   # 例 (49, 2, 64, 96)
    )
        # SDPA 输出: (B, nHead, nQuery, head_dim), 例 (49, 2, 16, 96)
        x = self.proj(x.transpose(1, 2).reshape(B, H, W, -1))
        # (B, H, W, dim_out), 例 (49, 4, 4, 192)
        return x


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride

        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=int(dim_out * mlp_ratio),
            out_features=dim_out,
            num_layers=2,
            act_layer=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x: (B, H, W, C)，例 blk3 = (1, 28, 28, 192)
        # 一个 block 的骨架 = 标准 Transformer block:
        #   残差1: x + Attn(Norm(x))     残差2: x + MLP(Norm(x))
        # 区别在于 Attn 是在"切出来的小窗口"内做的; 若本块是 stage 切换块, 还顺带做 Q-pooling 下采样。

        # 构造投影 残差连接
        shortcut = x  # B, H, W, C
        # 归一化
        x = self.norm1(x)

        if self.dim != self.dim_out:
            # ---- 仅 stage 切换块 (dim != dim_out) 走这里 ----
            # 残差分支要先对齐主分支的新形状: Linear 提升通道 + MaxPool 下采样。
            # 例 blk5: x (1,28,28,192) -> proj (1,28,28,384) -> pool (1,14,14,384)
            # 池化调整B, H, W,C中 HW 维度
            # nn.linear(x) 修改最后一个维度
            shortcut = do_pool(self.proj(x), self.pool)

        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            # 把整张特征图切成 (H/ws × W/ws) 个小窗口, 展平窗口成批维度:
            # (B, H, W, C) -> (B * Hn * Wn, ws, ws, C)
            # 例 blk3: (1,28,28,192) -> 49 个 (4,4,192) -> (49, 4, 4, 192)
            # 例 blk5: (1,28,28,192) -> 49 个 (4,4,192) -> (49, 4, 4, 192)
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        # 注意力后:
        #   普通块:   形状不变 (B*Hn*Wn, ws, ws, C)，例 (49,4,4,192)
        #   q-pool块: Q 被下采样, 每个窗口边长减半, 例 blk5 -> (49,2,2,384)
        if self.q_stride:
            # ---- 仅 stage 切换块: Q-pooling 后窗口边长也变了, 重算反分区参数 ----
            # Q 池化把每个窗口缩小了 q_stride 倍 → 反分区窗口 = 原窗口 / stride
            # 例 blk5: 4//2 = 2;  blk2: 8//2 = 4;  blk21: 14//2 = 7
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            # 反分区要还原到的目标尺寸 = 残差(已经下采样过)的尺寸
            H, W = shortcut.shape[1:3]
            # 例 blk5: shortcut (1,14,14,384) → H=W=14

            # 目标尺寸未必整除新窗口, 补零到能整除
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)  # 例 blk5: (14,14)

        # Reverse window partition
        if self.window_size > 0:
            # 把窗口拼回整张特征图, 裁掉 padding:
            # (B*Hn*Wn, ws, ws, C) -> (B, H', W', C) -> crop -> (B, H, W, C)
            # 例 blk5: (49,2,2,384) -> (1,14,14,384)  ← 空间降采样完成
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)  # 残差相加, 形状对齐 (例 (1,14,14,384))
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x