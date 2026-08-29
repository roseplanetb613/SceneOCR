"""DBNet 风格文本检测头（可微分二值化文本检测）。

原理（DBNet, arXiv:2202.10304）:
    传统文本检测要"分割 → 阈值化 → 找框"，阈值化不可导。
    DBNet 把"阈值"也做成一个网络输出（阈值图），
    用可微分二值化 B = sigmoid(k * (P - T)) 把概率图 P 和阈值图 T 合成二值图，
    整个检测流程就能端到端训练。

输入: FPN 多尺度特征（components.neck.FpnNeck 的输出，BCHW，通常 3 层）
    例 1024 输入 → 3 层 256ch 特征，分辨率 256/128/64（stride 4/8/16）
输出: 3 张图（都在 stride-4 分辨率，例 256×256）:
    prob   概率图 logits（文本区域的概率）
    thr    阈值图（可学习的二值化阈值）
    binary 二值图 = sigmoid(k * (prob - thr))，用于训练和最终检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DBNetHead(nn.Module):
    def __init__(
        self,
        in_chans: int = 256,   # FPN 每层通道数
        num_levels: int = 3,   # 使用的 FPN 层数
        inner_chans: int = 256,  # 融合后的特征通道
        k: float = 10.0,       # 可微分二值化的放大系数(小一点梯度更平缓, 训练更稳)
    ):
        super().__init__()
        self.k = k
        self.num_levels = num_levels

        # ---- 1. 每层一个 3×3 卷积统一通道 ----
        self.level_convs = nn.ModuleList(
            [nn.Conv2d(in_chans, inner_chans, 3, padding=1) for _ in range(num_levels)]
        )

        # ---- 2. 拼接后的融合卷积 ----
        # 用 GroupNorm 代替 BatchNorm: 小 batch + 训练骨干时 BN 统计不稳定
        # 会导致融合特征塌缩(概率图输出恒定 0.5), GroupNorm 与 batch 大小无关更稳
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(inner_chans * num_levels, inner_chans, 3, padding=1),
            nn.GroupNorm(32, inner_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_chans, inner_chans, 3, padding=1),
            nn.GroupNorm(32, inner_chans),
            nn.ReLU(inplace=True),
        )

        # ---- 3. 双分支：概率图 + 阈值图 ----
        self.prob_head = nn.Conv2d(inner_chans, 1, 3, padding=1)
        self.thr_head = nn.Conv2d(inner_chans, 1, 3, padding=1)

    def forward(self, features):
        """
        features: list of BCHW，按分辨率从大到小（stride 4/8/16）
        例 [(B,256,256,256), (B,256,128,128), (B,256,64,64)]
        """
        assert len(features) == self.num_levels
        target_hw = features[0].shape[-2:]  # 最大分辨率 = stride-4

        # ---- 融合：各层卷积 → 上采样到 stride-4 → 拼接 ----
        feats = []
        for conv, f in zip(self.level_convs, features):
            f = conv(f)
            if f.shape[-2:] != target_hw:
                f = F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False)
            feats.append(f)
        x = torch.cat(feats, dim=1)  # (B, 3*C, H/4, W/4)
        x = self.fuse_conv(x)        # (B, C, H/4, W/4)

        # ---- 概率图 + 阈值图 ----
        prob = self.prob_head(x)     # (B, 1, H/4, W/4) logits
        thr = self.thr_head(x)       # (B, 1, H/4, W/4)
        # 可微分二值化: k 放大 prob-thr 的差异, sigmoid 压到 (0,1)
        binary = torch.sigmoid(self.k * (prob - thr))  # (B, 1, H/4, W/4)

        return {"prob": prob, "thr": thr, "binary": binary}
