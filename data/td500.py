"""MSRA-TD500 文本检测数据集处理：解析旋转框标注 → 生成 DBNet 训练用的 GT 掩膜。

MSRA-TD500 .gt 每行: `index hard x y w h angle`
    (x,y) 是旋转矩形【中心】, w/h 是宽高, angle 是弧度。
    hard=1 表示难识别区域(标准做法: 训练时排除)。

DBNet 需要三张 GT 掩膜(都在训练分辨率):
    full_mask   完整文本多边形(填充)   → 监督二值图 BCE
    prob_mask   收缩后的文本区域       → 监督概率图 Dice
    thr_mask    边界带(膨胀-腐蚀)      → 监督阈值图 L1
"""

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.db_masks import make_db_masks
from data.det_augment import random_crop_containing_text, flip_horizontal


def parse_td500_gt(gt_path, drop_hard=True):
    """解析 .gt → 多边形列表 [(4,2) np.float32, ...]（原图坐标）。"""
    polys = []
    with open(gt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            hard = int(parts[1])
            cx, cy, w, h, theta = map(float, parts[2:7])
            if drop_hard and hard == 1:
                continue  # 排除难识别区域
            # 旋转矩形中心+宽高+弧度 → 4 角
            cos, sin = np.cos(theta), np.sin(theta)
            hw, hh = w / 2, h / 2
            poly = np.array(
                [
                    [cx - hw * cos + hh * sin, cy - hw * sin - hh * cos],
                    [cx + hw * cos + hh * sin, cy + hw * sin - hh * cos],
                    [cx + hw * cos - hh * sin, cy + hw * sin + hh * cos],
                    [cx - hw * cos - hh * sin, cy - hw * sin + hh * cos],
                ],
                dtype=np.float32,
            )
            polys.append(poly)
    return polys


def draw_polygon_mask(polys, h, w, scale_x=1.0, scale_y=1.0):
    """多边形列表 → 填充掩膜 (h, w) uint8, 坐标乘 (scale_x, scale_y)。"""
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polys:
        p = poly.astype(np.float32)
        p[:, 0] *= scale_x
        p[:, 1] *= scale_y
        cv2.fillPoly(mask, [p.astype(np.int32)], 1)
    return mask


def shrink_mask(mask, kernel=7):
    """腐蚀收缩(概率图 GT): 把文本区域往里缩, 让模型只关注核心区域。"""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    return cv2.erode(mask, k)


def boundary_mask(mask, erode_kernel=5, dilate_kernel=9):
    """边界带(阈值图 GT): 膨胀 - 腐蚀 → 只在文本边缘一圈是 1。"""
    ke = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_kernel, erode_kernel))
    kd = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel, dilate_kernel))
    dil = cv2.dilate(mask, kd)
    ero = cv2.erode(mask, ke)
    return (dil - ero).clip(0, 1)


class MSRATD500Dataset(Dataset):
    """扫描 train/test 目录, 返回 (image(3,H,W), full_mask, prob_mask, thr_mask)。"""

    def __init__(self, root, split="train", img_size=512):
        self.dir = os.path.join(root, split)
        self.img_size = img_size
        self.samples = []
        for gt in sorted(os.listdir(self.dir)):
            if gt.endswith(".gt"):
                img_path = os.path.join(self.dir, gt[:-3] + ".JPG")
                if os.path.exists(img_path):
                    self.samples.append((img_path, os.path.join(self.dir, gt)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]
        # 读图 (BGR→RGB)，保持 numpy + 原图坐标；裁剪/缩放由
        # random_crop_containing_text 统一处理(图和多边形同步变换)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        polys = parse_td500_gt(gt_path)
        # 随机裁剪含文本区域(对照 EastRandomCropData) + 翻转
        img, polys = random_crop_containing_text(
            img, polys, target_size=(self.img_size, self.img_size))
        img, polys = flip_horizontal(img, polys)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        sm, smask, tm, tmask = make_db_masks(polys, self.img_size, self.img_size)

        def to_t(m):
            return torch.from_numpy(m.astype(np.float32))[None]  # (1,H,W)

        return img_t, to_t(sm), to_t(smask), to_t(tm), to_t(tmask)
