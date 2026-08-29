"""合成检测数据生成器：整图 + 文本框（随机背景/位置/旋转/纹理），给 DBNet 预训练。

思路（同 SynthText 的简化版）:
    检测要"在图上找文本区域"。合成数据 = 随机背景 + 随机放的"文本块"矩形。
    关键是把文本块做得多样、逼真, 让模型学到"文本区域"的共性(块状、和背景对比、
    内部有纹理), 再迁移到真实数据(合成预训练 → 真实微调, 这是 DBNet 的标准配方)。

每张图随机:
    - 背景: 深/浅随机 + 噪声 + 模糊(梯度式背景)
    - 文本块: 1~6 个, 随机位置/大小/宽高比/旋转/明暗对比, 内部画横线模拟文字笔画
"""

import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.db_masks import make_db_masks


def _rotated_rect(cx, cy, w, h, theta_deg):
    """中心+宽高+角度(度) → 4 角多边形。"""
    theta = np.radians(theta_deg)
    cos, sin = np.cos(theta), np.sin(theta)
    hw, hh = w / 2, h / 2
    return np.array(
        [
            [cx - hw * cos + hh * sin, cy - hw * sin - hh * cos],
            [cx + hw * cos + hh * sin, cy + hw * sin - hh * cos],
            [cx + hw * cos - hh * sin, cy + hw * sin + hh * cos],
            [cx - hw * cos - hh * sin, cy - hw * sin + hh * cos],
        ],
        dtype=np.float32,
    )


class SynthDetDataset(Dataset):
    def __init__(self, num_samples=10000, img_size=512, seed=None,
                 boxes_range=(1, 6)):
        self.num_samples = num_samples
        self.img_size = img_size
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.boxes_range = boxes_range

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        S = self.img_size
        r = self.rng

        # ---- 随机背景 ----
        if r.random() < 0.5:
            bg = r.choice([(20, 25, 30), (35, 30, 25), (25, 20, 35)])
            text_color = (230, 230, 230)
        else:
            bg = r.choice([(235, 235, 235), (250, 245, 235), (235, 245, 250)])
            text_color = (30, 30, 30)
        img = np.full((S, S, 3), bg, dtype=np.uint8)
        # 噪声 + 轻微模糊(模拟拍摄)
        img = img.astype(np.float32)
        img += self.np_rng.normal(0, r.uniform(3, 12), img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # ---- 放置文本块 ----
        polys = []
        for _ in range(r.randint(*self.boxes_range)):
            w = r.randint(50, int(S * 0.6))
            h = r.randint(15, 60)  # 文本块: 宽 > 高
            cx, cy = r.randint(w // 2, S - w // 2), r.randint(h // 2, S - h // 2)
            theta = r.uniform(-20, 20)  # 轻旋转
            poly = _rotated_rect(cx, cy, w, h, theta)
            # 填充文本块颜色(和背景对比)
            fg = tuple(int(c) for c in text_color)
            cv2.fillPoly(img, [poly.astype(np.int32)], fg)
            # 块内部画几条横线模拟文字笔画(和背景色形成内部纹理)
            for k in range(max(2, h // 8)):
                yy = int(cy - h / 3 + k * (h / 3) / max(h // 8, 1))
                pts = np.array([[cx - w * 0.4, yy], [cx + w * 0.4, yy]], dtype=np.int32)
                cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, bg, 2)
            polys.append(poly)

        # ---- GT 掩膜 (收缩图/掩膜/软阈值图/阈值掩膜) ----
        sm, smask, tm, tmask = make_db_masks(polys, S, S)

        def to_t(m):
            return torch.from_numpy(m.astype(np.float32))[None]

        img_t = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        return img_t, to_t(sm), to_t(smask), to_t(tm), to_t(tmask)
