"""SynthText 检测数据集：读预处理缓存 + 图，生成 DBNet GT 掩膜。

用法: 先跑 data/synthtext_preprocess.py 生成缓存, 再实例化本数据集。
"""

import os
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.db_masks import make_db_masks
from data.det_augment import random_crop_containing_text, flip_horizontal


class SynthTextDataset(Dataset):
    def __init__(self, root, cache="data/cache_synthtext.pkl",
                 img_size=512, max_images=None, seed=None):
        self.root = root
        self.img_size = img_size
        with open(cache, "rb") as f:
            data = pickle.load(f)
        names, polys = data["names"], data["polys"]
        if max_images is not None:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(names), size=min(max_images, len(names)), replace=False)
            idx = np.sort(idx)
            names = [names[i] for i in idx]
            polys = [polys[i] for i in idx]
        self.names, self.polys = names, polys

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = os.path.join(self.root, name)
        # cv2.imread 不支持中文路径 → 用 np.fromfile + imdecode(Unicode 安全)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            # 个别图缺失(缓存兜底): 黑图 + 空 GT(极少数损坏图, 可接受)
            img = np.zeros((self.img_size, self.img_size, 3), np.uint8)
            polys = []
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            polys = self.polys[idx]
        # 随机裁剪一个【含文本】的区域(对照 EastRandomCropData) + 水平翻转
        img, polys = random_crop_containing_text(
            img, polys, target_size=(self.img_size, self.img_size))
        img, polys = flip_horizontal(img, polys)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)

        # img 已是 img_size×img_size, polys 已同步缩放 → scale=1
        sm, smask, tm, tmask = make_db_masks(polys, self.img_size, self.img_size)

        def to_t(m):
            return torch.from_numpy(m.astype(np.float32))[None]

        return img_t, to_t(sm), to_t(smask), to_t(tm), to_t(tmask)
