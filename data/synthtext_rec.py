"""SynthText 识别数据管线：wordBB 词框 → 词级裁剪图 + 标签。

数据现实: 本机这份 SynthText 的 txt/charBB 组织混乱(多词合并/空格填充/换行,
对齐率仅 ~17%)。因此采用最稳的策略:
    1. 只取 txt 与 wordBB **1:1 词级对齐** 的图;
    2. 词标签严格过滤: 无换行/内部空格/标点, 纯 [A-Za-z0-9], 长度 1~10;
    3. 缓存 (图路径, 词框, 标签) 轻量元组, 训练时在线裁剪(磁盘友好)。

用法:
    python data/synthtext_rec.py --root <SynthText> --cache data/cache_synthtext_rec.pkl
"""

import argparse
import os
import pickle

import cv2
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

IMG_H, IMG_W = 32, 128


# ==================== 提取(离线, 一次性) ====================
def _clean(text):
    """返回干净词或 None。要求: 无换行/无内部空格/纯字母数字/长度 1~10。"""
    s = str(text).strip()
    if len(s) < 1 or len(s) > 10:
        return None
    if "\n" in s or " " in s:
        return None
    if not s.isalnum():
        return None
    return s


def extract(root, max_words=None, seed=None):
    """从 gt.mat 提取 [(img_path, box(2,4), word)] 列表。"""
    gt_path = os.path.join(root, "gt.mat")
    m = sio.loadmat(gt_path, variable_names=["wordBB", "txt", "imnames"])
    wordBB = m["wordBB"][0]
    txt = m["txt"][0]
    imnames = m["imnames"][0]

    items = []
    skipped = {"not_1to1": 0, "dirty": 0, "missing": 0}
    for i in range(len(imnames)):
        name = str(imnames[i][0])
        t, wb = txt[i], wordBB[i]
        if t is None or len(t) == 0 or wb is None:
            continue
        n_words = wb.shape[2] if wb.ndim == 3 else 1
        if len(t) != n_words:
            skipped["not_1to1"] += 1
            continue
        if not os.path.exists(os.path.join(root, name)):
            skipped["missing"] += 1
            continue
        img_path = name
        for j in range(n_words):
            word = _clean(t[j])
            if word is None:
                skipped["dirty"] += 1
                continue
            box = wb[:, :, j] if wb.ndim == 3 else wb  # (2,4) x/y 两行
            items.append((img_path, box.astype(np.float32), word))
        if max_words is not None and len(items) >= max_words:
            break
    print(f"提取完成: {len(items)} 词 (跳过: {skipped})")
    if seed is not None:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(items))
        items = [items[k] for k in idx]
    return items


# ==================== 数据集(训练时在线裁剪) ====================
class SynthTextRecDataset(Dataset):
    """返回 (img(3,32,128) 归一化, text)。裁剪: wordBB 四点 → 透视矫正到画布。"""

    def __init__(self, root, cache="data/cache_synthtext_rec.pkl",
                 img_h=IMG_H, img_w=IMG_W, max_items=None, seed=None,
                 aug=False):
        self.root = root
        self.img_h, self.img_w = img_h, img_w
        self.aug = aug
        with open(cache, "rb") as f:
            self.items = pickle.load(f)
        if max_items is not None:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(self.items), size=min(max_items, len(self.items)),
                             replace=False)
            self.items = [self.items[k] for k in idx]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def _crop(self, img_path, box):
        """wordBB 四点 → 透视矫正到 (img_h, img_w)。"""
        img = cv2.imdecode(np.fromfile(os.path.join(self.root, img_path),
                                       dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        src = np.array([[box[0, j], box[1, j]] for j in range(4)], dtype=np.float32)
        # 合法性: 四点在图像内
        if (src[:, 0].min() < 0 or src[:, 0].max() > img.shape[1] or
                src[:, 1].min() < 0 or src[:, 1].max() > img.shape[0]):
            return None
        dst = np.array([[0, 0], [self.img_w - 1, 0],
                        [self.img_w - 1, self.img_h - 1], [0, self.img_h - 1]],
                       dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, M, (self.img_w, self.img_h),
                                   flags=cv2.INTER_CUBIC)

    def __getitem__(self, idx):
        img_path, box, word = self.items[idx]
        for _ in range(3):  # 坏图重试
            img = self._crop(img_path, box)
            if img is not None:
                break
        if img is None:
            img = np.zeros((self.img_h, self.img_w, 3), np.uint8)
        if self.aug:
            # 适合单词识别的增强: 旋转/亮度/噪声(不用翻转, 翻转会反转字序但标签不变)
            if self.rng.random() < 0.5:
                ang = self.rng.uniform(-8, 8)
                M2 = cv2.getRotationMatrix2D((self.img_w / 2, self.img_h / 2), ang, 1.0)
                img = cv2.warpAffine(img, M2, (self.img_w, self.img_h),
                                     flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
            img = np.clip(img.astype(np.float32) * self.rng.uniform(0.7, 1.3), 0, 255)
            img = img.astype(np.uint8)
            if self.rng.random() < 0.3:
                noise = np.random.default_rng().normal(0, 5, img.shape)
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        t = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        return t, word


# ==================== 预裁剪缓存(离线, 训练时零读图开销) ====================
def gen_precrop(root, cache="data/cache_synthtext_rec.pkl", max_items=None,
                out_crops="data/crops_synthtext_rec.npy", out_labels="data/labels_synthtext_rec.pkl",
                img_h=IMG_H, img_w=IMG_W, seed=None, verbose=True):
    """一次性把词框裁剪成 (N,3,H,W) uint8 + 标签列表, 存盘。训练直接读内存。"""
    import time
    ds = SynthTextRecDataset(root, cache=cache, max_items=max_items, seed=seed, aug=False)
    n = len(ds)
    crops = np.zeros((n, 3, img_h, img_w), np.uint8)
    labels = []
    t0 = time.time()
    for i in range(n):
        img, word = ds[i]
        crops[i] = (img.numpy() * 255).astype(np.uint8)
        labels.append(word)
        if verbose and (i + 1) % 10000 == 0:
            print(f"  {i+1}/{n} ({time.time()-t0:.0f}s)")
    np.save(out_crops, crops)
    with open(out_labels, "wb") as f:
        pickle.dump(labels, f)
    print(f"预裁剪完成: {n} 词, {out_crops} ({os.path.getsize(out_crops)/1e6:.0f} MB)")


class PrecroppedRecDataset(Dataset):
    """从预裁剪缓存加载 (N,3,32,128) uint8, 训练时零读图/裁剪开销。"""

    def __init__(self, crops_path="data/crops_synthtext_rec.npy",
                 labels_path="data/labels_synthtext_rec.pkl", aug=False, seed=None,
                 img_h=IMG_H, img_w=IMG_W):
        import numpy as np
        self.crops = np.load(crops_path)  # (N,3,H,W) uint8
        with open(labels_path, "rb") as f:
            self.labels = pickle.load(f)
        self.aug = aug
        self.img_h, self.img_w = img_h, img_w
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.crops[idx]  # (3,H,W) uint8
        if self.aug:
            arr = img.transpose(1, 2, 0).copy()  # HWC
            if self.rng.random() < 0.5:
                ang = self.rng.uniform(-8, 8)
                M = cv2.getRotationMatrix2D((self.img_w / 2, self.img_h / 2), ang, 1.0)
                arr = cv2.warpAffine(arr, M, (arr.shape[1], arr.shape[0]),
                                     flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
            arr = np.clip(arr.astype(np.float32) * self.rng.uniform(0.7, 1.3), 0, 255)
            if self.rng.random() < 0.3:
                arr = arr + np.random.default_rng().normal(0, 5, arr.shape)
                arr = np.clip(arr, 0, 255)
            img = arr.astype(np.uint8).transpose(2, 0, 1)
        t = torch.from_numpy(img).float().div(255.0)
        return t, self.labels[idx]


# ==================== 字符集统计 ====================
def charset_stats(items, top=70):
    from collections import Counter
    c = Counter("".join(w for _, _, w in items))
    print(f"词数: {len(items)}, 字符种类: {len(c)}")
    for ch, n in c.most_common(top):
        print(f"  {repr(ch)}: {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="E:/迅雷下载/SynthText/SynthText")
    ap.add_argument("--cache", default="data/cache_synthtext_rec.pkl")
    ap.add_argument("--max_words", type=int, default=300000)
    args = ap.parse_args()

    items = extract(args.root, max_words=args.max_words, seed=42)
    os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
    with open(args.cache, "wb") as f:
        pickle.dump(items, f)
    print(f"缓存已存: {args.cache} ({os.path.getsize(args.cache)/1e6:.1f} MB)")
    charset_stats(items)
