"""DropPath：随机深度 (Stochastic Depth) —— 训练时按样本整条路径丢弃。"""

import torch
import torch.nn as nn


class DropPath(nn.Module):
    # adapted from timm layers/drop.py
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x  # 推理时恒等
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 每样本一个掩码, 非逐元素
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)  # 保留时放大, 保持期望不变
        return x * random_tensor
