"""检测后处理：概率图 → 文本框。

流程: 概率图 → 阈值二值化 → 连通域 → 每个连通域的外接框。
连通域用 scipy.ndimage（若可用）；否则退回"整张前景一个框"的简化版。
"""

import numpy as np


def prob_to_boxes(prob, thr=0.3, min_area=16):
    """
    prob: (H, W) 概率图（0~1）或 logits
    thr:  二值化阈值
    返回: list of [x1, y1, x2, y2]（像素坐标，在 prob 分辨率下）
    """
    p = np.asarray(prob)
    if p.max() > 1.5:  # 是 logits → 过 sigmoid
        p = 1.0 / (1.0 + np.exp(-p))
    binary = p > thr

    if not binary.any():
        return []

    try:
        from scipy import ndimage

        labels, n = ndimage.label(binary)
        boxes = []
        for i in range(1, n + 1):
            ys, xs = np.where(labels == i)
            if len(xs) < min_area:
                continue
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
        return boxes
    except ImportError:
        # 没有 scipy: 简化, 整张前景一个框
        ys, xs = np.where(binary)
        return [[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]]
