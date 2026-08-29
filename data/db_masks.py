"""DBNet GT 掩膜生成（对照 PaddleOCR_DBNet 原版实现）。

生成 4 张监督图:
    shrink_map      收缩后的文本多边形(填充 1)   → 概率图 BCE + binary 图 Dice 的目标
    shrink_mask     有效区域掩膜(太小/忽略的框置 0) → 只监督有效文本
    threshold_map   软阈值图(0.3~0.7, 边界近的亮)  → 阈值图 L1 的目标
    threshold_mask  膨胀区域掩膜(只在这圈里监督阈值)
"""

import cv2
import numpy as np


def shrink_polygon(polygon, ratio=0.4):
    """多边形向质心收缩 ratio 倍(近似 Vatti 收缩, 四点多边形够用)。"""
    cx, cy = polygon[:, 0].mean(), polygon[:, 1].mean()
    return np.column_stack(
        [cx + (polygon[:, 0] - cx) * ratio, cy + (polygon[:, 1] - cy) * ratio]
    ).astype(np.float32)


def _expand_mask(mask, kernel=9):
    """膨胀掩膜(近似向外扩一圈, 用于阈值图区域)。"""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    return cv2.dilate(mask, k)


def make_db_masks(polys, h, w, scale_x=1.0, scale_y=1.0,
                  min_text_size=8, shrink_ratio=0.4,
                  thresh_min=0.3, thresh_max=0.7):
    """
    polys: 原图坐标多边形列表 (N,4,2)
    返回 (shrink_map, shrink_mask, threshold_map, threshold_mask)，都在 (h,w)。
    """
    shrink_map = np.zeros((h, w), np.float32)
    shrink_mask = np.ones((h, w), np.float32)
    threshold_map = np.zeros((h, w), np.float32)
    threshold_mask = np.zeros((h, w), np.float32)

    for poly in polys:
        p = poly.astype(np.float32).copy()
        p[:, 0] *= scale_x
        p[:, 1] *= scale_y
        hgt = max(p[:, 1]) - min(p[:, 1])
        wd = max(p[:, 0]) - min(p[:, 0])
        if min(hgt, wd) < min_text_size:
            # 太小的文本: 从 shrink_mask 里排除(不监督), 跳过
            cv2.fillPoly(shrink_mask, [p.astype(np.int32)], 0.0)
            continue

        # 收缩图(概率图/binary 的目标)
        shrunk = shrink_polygon(p, shrink_ratio)
        if shrunk.shape[0] >= 3:
            cv2.fillPoly(shrink_map, [shrunk.astype(np.int32)], 1.0)

        # 阈值图: 在膨胀区域内, 值 = 0.3~0.7, 越靠文本边界越接近 0.7
        full = np.zeros((h, w), np.uint8)
        cv2.fillPoly(full, [p.astype(np.int32)], 1)
        # 膨胀量近似按文本尺寸缩放
        dist_px = max(int(max(hgt, wd) * (1 - shrink_ratio) * 0.5), 3)
        expanded = _expand_mask(full, kernel=max(dist_px * 2 + 1, 3))
        inside = expanded > 0
        dist = cv2.distanceTransform(expanded, cv2.DIST_L2, 3)
        maxd = dist[inside].max() if inside.any() else 1.0
        # 边界处 dist 小 → 值大(接近 0.7); 中心 dist 大 → 值小(接近 0.3)
        val = 1.0 - dist / maxd
        val = thresh_min + val * (thresh_max - thresh_min)
        threshold_map = np.where(inside, np.maximum(threshold_map, val), threshold_map)
        threshold_mask = np.maximum(threshold_mask, expanded.astype(np.float32))

    return shrink_map, shrink_mask, threshold_map, threshold_mask
