"""CXBlock：ConvNeXt 块（通用 CNN 模块，OCR 里可作特征精修/检测头的基本块）。

自实现的 ConvNeXt 风格 CNN 块。
结构: 深度卷积 → 通道维 LayerNorm → 升维 Linear(×4) → GELU → 降维 Linear → LayerScale → 残差
"""

import torch
import torch.nn as nn

from components.blocks.drop_path import DropPath
from components.blocks.norm import LayerNorm2d


class CXBlock(nn.Module):
    """ConvNeXt Block（用 Linear 实现 1×1 卷积）。

    例 (B, C, H, W): dwconv → LN → permute(B,H,W,C) → Linear(C→4C) → GELU
        → Linear(4C→C) → gamma 缩放 → permute 回 → + 输入
    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=1e-6,  # Layer Scale 初始值: 小 → 开始时几乎只走残差
        use_dwconv=True,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,  # depthwise conv: 每通道独立
        )
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise(1x1) conv 用 Linear 实现
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x
