"""对比两种裁剪: 轴对齐外接矩形 vs 透视矫正(wordBB 4点 → 128x32), 可视化保存。"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

root = "E:/迅雷下载/SynthText/SynthText"
cache = "data/cache_synthtext_rec.pkl"
os.makedirs("outputs/debug_rec", exist_ok=True)

import pickle
with open(cache, "rb") as f:
    items = pickle.load(f)
print("缓存词数:", len(items))

H, W = 32, 128


def crop_axis(img, box):
    xs, ys = box[0], box[1]
    x1, x2 = max(int(xs.min()), 0), min(int(xs.max()), img.shape[1])
    y1, y2 = max(int(ys.min()), 0), min(int(ys.max()), img.shape[0])
    crop = img[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    scale = min(H / h, W / w)
    nh, nw = max(int(h * scale), 1), max(int(w * scale), 1)
    canvas = np.zeros((H, W, 3), np.uint8)
    r = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)
    ox, oy = (W - nw) // 2, (H - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = r
    return canvas


def crop_persp(img, box):
    src = np.array([[box[0, j], box[1, j]] for j in range(4)], dtype=np.float32)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC)


for idx in [0, 100, 500, 1000, 2000, 5000]:
    img_path, box, word = items[idx]
    img = cv2.imdecode(np.fromfile(os.path.join(root, img_path), dtype=np.uint8),
                       cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 原图 + 词框
    vis = img.copy()
    pts = np.array([[box[0, j], box[1, j]] for j in range(4)], dtype=np.int32)
    cv2.polylines(vis, [pts], True, (255, 0, 0), 3)
    # 放大对比图 (3x)
    ca, cp = crop_axis(img, box), crop_persp(img, box)
    scale = 4
    grid = np.zeros((H * scale * 2 + 10, W * scale * 2 + 10, 3), np.uint8)
    grid[:H * scale, :W * scale] = cv2.resize(ca, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    grid[H * scale + 10:, :W * scale] = cv2.resize(cp, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    # 画标签
    cv2.putText(grid, word, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(grid, "axis / persp", (W * scale // 2, H * scale * 2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(f"outputs/debug_rec/{idx}_{word}.png",
                cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"保存 {idx}_{word}: 词框中心=({box[0].mean():.0f},{box[1].mean():.0f})")
print("完成")
