"""PositionEmbeddingSine：二维正弦位置编码（给 FpnNeck 的输出生成位置编码）。

自实现的正弦位置编码（PositionEmbeddingSine），对二维特征图生成网格位置编码。
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    二维版"Attention Is All You Need"位置编码，推广到图像。

    作用：给特征图上每个位置 (y, x) 生成一个维度为 num_pos_feats 的向量，
    编码"这个 token 在图上处于什么位置"，让后续注意力能感知空间结构。

    原理（无参数，纯坐标 + 公式，不参与训练）：
      - 对 x 坐标和 y 坐标各算一组不同频率的正弦/余弦：
          低频通道 → 编码"大致在哪个区域"
          高频通道 → 编码"精确到哪个像素"
      - 每个频率输出 sin 和 cos 两个值（交替排布），所以
        坐标维通道 = 频率数 × 2 = num_pos_feats（例 256）
      - x、y 两组拼起来，再加归一化使编码与分辨率无关（归一化到 [0, 2π)）。

    例：输入特征 (B, 256, 56, 56) → 输出 (B, 256, 56, 56)，通道一一对应相加即可用。
    """

    def __init__(
        self,
        num_pos_feats,  # 期望的输出通道数，必须为偶数；本项目 = d_model = 256
        temperature: int = 10000,  # 频率衰减基，越大低频越多
        normalize: bool = True,  # 归一化坐标 → 位置编码与分辨率无关
        scale: Optional[float] = None,  # 坐标放大倍数，默认 2π（正弦一个周期）
        # 下面参数只用于预热缓存，方便编译加速，与功能无关
        warmup_cache: bool = True,
        image_size: int = 1024,
        strides: Tuple[int] = (4, 8, 16, 32),
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        # 每个坐标 (x 或 y) 分一半通道，另一半由 sin/cos 成对补足
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi  # 归一化后坐标范围就是 [0, 2π]
        self.scale = scale

        # 按 (H, W) 缓存已生成的位置编码，避免重复计算
        self.cache = {}
        if warmup_cache and torch.cuda.is_available():
            # 预先为常见分辨率(1024/stride)在 cuda 上算好，帮助编译
            device = torch.device("cuda")
            for stride in strides:
                cache_key = (image_size // stride, image_size // stride)
                self._pe(1, device, *cache_key)

    @torch.no_grad()
    def _pe(self, B, device, *cache_key):
        # 生成单张 (B=1) 特征图的位置编码，形状 (1, num_pos_feats, H, W)
        H, W = cache_key
        if cache_key in self.cache:
            # 命中缓存: 取回单张编码 → 补 batch 维 → 沿 batch 复制 B 份
            return self.cache[cache_key].to(device)[None].repeat(B, 1, 1, 1)

        # ---- 1. 构造坐标网格 ----
        # y_embed (B, H, W): 每个位置的"行号"，从 1 开始
        y_embed = (
            torch.arange(1, H + 1, dtype=torch.float32, device=device)
            .view(1, -1, 1)  # (1, H, 1)
            .repeat(B, 1, W)  # (B, H, W)
        )
        # x_embed (B, H, W): 每个位置的"列号"，从 1 开始
        x_embed = (
            torch.arange(1, W + 1, dtype=torch.float32, device=device)
            .view(1, 1, -1)  # (1, 1, W)
            .repeat(B, H, 1)  # (B, H, W)
        )

        # ---- 2. 归一化到 [0, 2π) ----
        if self.normalize:
            eps = 1e-6
            # 除以最大坐标(最后一行/列)，把坐标压到 [0,1] 再乘 2π
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # ---- 3. 构造频率序列 ----
        # dim_t: 每条通道的频率分母，指数衰减 → 从低频到高频
        # 例 num_pos_feats=128: dim_t = 10000^(0, 0, 1/128, 1/128, 2/128, ...)
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # ---- 4. 对 x、y 各自编码: 位置 / 频率 → sin/cos ----
        # pos_x (B, H, W, num_pos_feats): 列号在每个频率下的正弦值
        pos_x = x_embed[:, :, :, None] / dim_t  # 广播: (B,H,W,1) / (num_pos_feats,)
        pos_y = y_embed[:, :, :, None] / dim_t
        # 偶数频通道取 sin，奇数频通道取 cos，再交替拼起来:
        # (B,H,W,num_pos_feats/2) sin + (B,H,W,num_pos_feats/2) cos
        # → stack 成 (..., num_pos_feats/2, 2) → flatten 回 (..., num_pos_feats)
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        # ---- 5. y 编码 + x 编码 → (B, 2*num_pos_feats, H, W) = (B, num_pos_feats, H, W) ----
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]  # 缓存单张（去掉 batch 维）
        return pos

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        # 输入 x: (B, C, H, W)，只需用到它的空间尺寸
        B = x.shape[0]
        cache_key = (x.shape[-2], x.shape[-1])
        return self._pe(B, x.device, *cache_key)  # (B, num_pos_feats, H, W)


# =============================================================================
# PositionEmbeddingRandom：随机频率位置编码（给点/框这类稀疏坐标用）
# 从 prompt_encoder/position_embedding_random.py 并入。与 PositionEmbeddingSine
# （固定频率、适合密集网格）互补：这个用随机高斯频率，适合稀疏点坐标。
# =============================================================================

import numpy as np

from typing import Any


class PositionEmbeddingRandom(nn.Module):
    """随机频率位置编码：坐标(归一化到[-1,1])乘随机高斯矩阵 → 高维 → sin/cos。"""

    def __init__(self, num_pos_feats: int = 64, scale: float = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        # 随机频率矩阵 (2, F)，2 行对应 x/y；训练中固定(buffer 不进优化器)
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        # coords 归一化到 [0,1] → [-1,1] → @ 随机矩阵 → 2π → sin/cos
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """给整张 (h,w) 网格生成密集位置编码 (C, H, W)。"""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))  # (h,w,2F)
        return pe.permute(2, 0, 1)  # (C, H, W)

    def forward_with_coords(self, coords_input, image_size) -> torch.Tensor:
        """像素坐标 → 归一化 → 编码，输出 (B, N, 2F)。"""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))
