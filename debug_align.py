"""数值验证 SynthText/TD500 的 GT 掩膜与图像是否对齐(不用 GPU)。

思路: 文本区域内部应有明显笔画/边缘结构。
  - 在 shrink_map 内部像素上, 图像的局部梯度幅度(边缘强度)应显著高于背景。
  - 若掩膜错位(坐标未同步变换), 该统计会消失或反转。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

from data.synthtext import SynthTextDataset
from data.td500 import MSRATD500Dataset


def edge_stats(img_np, mask):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    inside = mask > 0.5
    outside = mask <= 0.5
    if inside.sum() < 10 or outside.sum() < 10:
        return None
    return mag[inside].mean(), mag[outside].mean(), inside.sum() / mask.size


def check(ds, name, n=8):
    print(f"\n===== {name} 对齐检查 (前 {n} 张) =====")
    ok = 0
    for i in range(n):
        img, sm, smask, tm, tmask = ds[i]
        if isinstance(img, np.ndarray):
            img_np = img
        else:
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask = sm[0].numpy() if not isinstance(sm, np.ndarray) else sm
        r = edge_stats(img_np, mask)
        if r is None:
            print(f"  [{i}] 掩膜为空, 跳过")
            continue
        inside, outside, frac = r
        ratio = inside / max(outside, 1e-6)
        good = ratio > 1.3
        ok += good
        print(f"  [{i}] 文本区边缘强度={inside:7.1f} 背景区={outside:7.1f} "
              f"比值={ratio:5.2f} 前景占比={frac*100:5.1f}% {'OK' if good else 'MISALIGNED?'}")
    print(f"  → {ok}/{n} 张对齐正常")


check(SynthTextDataset("E:/迅雷下载/SynthText/SynthText",
                       cache="data/cache_synthtext.pkl",
                       img_size=640, max_images=20000, seed=1), "SynthText")

td500_root = "C:/Users/Inspiration/Desktop/MSRA-TD500"
if os.path.isdir(td500_root):
    check(MSRATD500Dataset(td500_root, split="train", img_size=512), "MSRA-TD500 train")
else:
    print(f"\nTD500 数据集不存在: {td500_root}, 跳过")
