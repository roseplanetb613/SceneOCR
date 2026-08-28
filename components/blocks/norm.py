"""LayerNorm2d：沿通道维做 LayerNorm（对 (N, C, H, W) 特征图）。"""

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """在通道维上归一化，等价于把 H,W 展平后的逐 token LayerNorm。"""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)  # (N,1,H,W)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
