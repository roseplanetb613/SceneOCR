"""检测数据增强（对照 PaddleOCR_DBNet 的 EastRandomCropData + Fliplr）。

核心: 随机裁剪一个【包含文本】的区域(最多试 max_tries 次), 再缩放到 target_size。
这保证每个训练样本里都有文本, 不会出现"大半是背景"的样本把模型带偏。
"""

import cv2
import numpy as np


def _split_regions(axis):
    regions = []
    start = 0
    for i in range(1, axis.shape[0]):
        if axis[i] != axis[i - 1] + 1:
            regions.append(axis[start:i])
            start = i
    regions.append(axis[start:])
    return regions


def _poly_in_rect(poly, x, y, w, h):
    return not (poly[:, 0].max() < x or poly[:, 0].min() > x + w
                or poly[:, 1].max() < y or poly[:, 1].min() > y + h)


def random_crop_containing_text(img, polys, target_size=(640, 640),
                                max_tries=50, min_side_ratio=0.1):
    """从 img 里随机裁一块含文本的区域, 缩放到 target_size(keep ratio + pad)。
    返回 (cropped_img, scaled_polys)，polys 坐标已同步缩放。
    """
    h, w = img.shape[:2]
    # 标记文本占用的行列
    h_arr = np.zeros(h, np.int32)
    w_arr = np.zeros(w, np.int32)
    for p in polys:
        p = np.round(p).astype(np.int32)
        w_arr[max(p[:, 0].min(), 0):min(p[:, 0].max(), w)] = 1
        h_arr[max(p[:, 1].min(), 0):min(p[:, 1].max(), h)] = 1
    h_axis = np.where(h_arr == 0)[0]
    w_axis = np.where(w_arr == 0)[0]

    crop = None
    if len(h_axis) > 0 and len(w_axis) > 0:
        h_regions = _split_regions(h_axis)
        w_regions = _split_regions(w_axis)
        for _ in range(max_tries):
            def pick(regions, axis):
                if len(regions) > 1:
                    r = regions[np.random.randint(len(regions))]
                    return int(r[np.random.randint(len(r))])
                return int(np.random.choice(axis))
            x1, x2 = sorted([pick(w_regions, w_axis), pick(w_regions, w_axis)])
            y1, y2 = sorted([pick(h_regions, h_axis), pick(h_regions, h_axis)])
            if (x2 - x1) < min_side_ratio * w or (y2 - y1) < min_side_ratio * h:
                continue
            if any(_poly_in_rect(p, x1, y1, x2 - x1, y2 - y1) for p in polys):
                crop = (x1, y1, x2 - x1, y2 - y1)
                break
    if crop is None:
        crop = (0, 0, w, h)
    x, y, cw, ch = crop

    # 缩放 + 等比 pad 到 target_size
    scale = min(target_size[0] / cw, target_size[1] / ch)
    nh, nw = int(ch * scale), int(cw * scale)
    tw, th = target_size
    pad_img = np.zeros((th, tw, 3), img.dtype)
    resized = cv2.resize(img[y:y + ch, x:x + cw], (nw, nh))
    pad_img[:nh, :nw] = resized

    # 同步缩放文本多边形
    polys_scaled = []
    for p in polys:
        p2 = (p - (x, y)) * scale
        if not _poly_in_rect(p2, 0, 0, nw, nh):
            continue  # 裁掉落在区域外的
        polys_scaled.append(p2.astype(np.float32))
    return pad_img, polys_scaled


def flip_horizontal(img, polys, p=0.5):
    """水平翻转(对照 Fliplr p=0.5)。"""
    if np.random.rand() < p:
        img = img[:, ::-1].copy()
        w = img.shape[1]
        polys = [np.column_stack([w - p[:, 0], p[:, 1]]).astype(np.float32)
                 for p in polys]
    return img, polys
