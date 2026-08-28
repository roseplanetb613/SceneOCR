"""合成文本行数据生成器：模拟真实 OCR 裁剪的文本行图，用于预训练。

思路（和 SynthText / MJSynth 类似）: 用渲染引擎生成"假的但像真的"训练数据,
量可以无限大、标签精确无误, 先让模型学会读字, 再在真实数据上微调。

每张图随机组合:
    - 字体 (多选几款 Windows TTF, 覆盖不同字形)
    - 背景色 (深底浅字 / 浅底深字 / 彩色)
    - 文字颜色 / 字号 (自适应缩放到适应宽度)
    - 轻微旋转 / 横向偏移
    - 高斯噪声 / 模糊 (模拟低质量拍摄)
"""

import os
import random

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# 精选可靠字体（C:/Windows/Fonts 里很多字体渲染会卡死/挂起，不能用全量）
_CURATED_FONTS = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/verdana.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/georgia.ttf",
]


def load_windows_fonts():
    """返回精选的可靠字体列表（逐个验证存在 + 能加载）。"""
    keep = []
    for p in _CURATED_FONTS:
        if not os.path.exists(p):
            continue
        try:
            f = ImageFont.truetype(p, 20)
            f.getbbox("ABC0129")
            keep.append(p)
        except Exception:
            pass
    if not keep:
        raise RuntimeError("精选字体全部不可用, 请检查 C:/Windows/Fonts")
    return keep


class SynthTextLineGenerator:
    """一次渲染一条文本行 → (image_tensor(3,H,W), text)。

    aug_level 控制增强强度:
        0 = 最小: 单字体 + 白字黑底 + 无旋转/噪声/模糊（用于验证管线能学）
        1 = 轻:   少量字体 + 明暗底随机 + 无噪声模糊
        2 = 全量: 多字体 + 明暗随机 + 旋转 + 噪声 + 模糊（默认, 贴近真实拍摄）
    """

    def __init__(self, img_h=32, img_w=128, font_paths=None, seed=None, aug_level=2):
        self.img_h, self.img_w = img_h, img_w
        self.aug_level = aug_level
        if aug_level == 0:
            # 最小增强: 固定用第一个字体
            font_paths = load_windows_fonts()[:1]
        self.font_paths = font_paths or load_windows_fonts()
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)  # 噪声用 numpy RNG
        if not self.font_paths:
            raise RuntimeError("找不到任何字体, 请指定 font_paths")

    def _pick_colors(self):
        """随机背景/前景色对。50% 深底浅字, 50% 浅底深字。"""
        if self.rng.random() < 0.5:
            bg = self.rng.choice([(25, 25, 30), (40, 40, 45), (10, 30, 20), (35, 25, 10)])
            fg = self.rng.choice([(240, 240, 240), (255, 220, 200), (200, 240, 255)])
        else:
            bg = self.rng.choice([(235, 235, 235), (250, 245, 235), (230, 245, 250)])
            fg = self.rng.choice([(20, 20, 20), (40, 30, 20), (15, 30, 40)])
        return bg, fg

    def _render_text(self, text, font_path, fg):
        """渲染文字到透明小图, 自适应字号塞进画布。"""
        pad = 4
        max_w, max_h = self.img_w - pad * 2, self.img_h - pad * 2
        for size in [24, 22, 20, 18, 16, 14]:
            font = ImageFont.truetype(font_path, size)
            bbox = font.getbbox(text)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if tw <= max_w and th <= max_h:
                break
        img = Image.new("RGBA", (max_w, max_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        ox, oy = self.rng.randint(0, max_w - tw), self.rng.randint(0, max_h - th)
        d.text((ox - bbox[0], oy - bbox[1]), text, fill=fg + (255,), font=font)
        return img

    def generate(self, text):
        """渲染给定 text, 返回 (torch.Tensor (3,H,W) 归一化, text)。"""
        aug = self.aug_level

        # ---- 字体 / 颜色 ----
        layer = None
        for _ in range(10):
            if aug == 0:
                font_path = self.font_paths[0]
                bg, fg = (0, 0, 0), (255, 255, 255)   # 固定白字黑底
            else:
                font_path = self.rng.choice(self.font_paths)
                bg, fg = self._pick_colors()
            try:
                layer = self._render_text(text, font_path, fg)
                break
            except Exception:
                continue
        if layer is None:
            raise RuntimeError(f"10 个字体都无法渲染 '{text}'")

        # ---- 旋转 (aug>=2 才加) ----
        if aug >= 2 and self.rng.random() < 0.6:
            angle = self.rng.uniform(-5, 5)
            layer = layer.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))

        # ---- 合成到背景 ----
        canvas = Image.new("RGB", (self.img_w, self.img_h), bg)
        canvas.paste(layer, (0, 0), layer)
        img = canvas

        # ---- 噪声 + 模糊 (aug>=2 才加) ----
        if aug >= 2:
            arr = np.array(img).astype(np.float32)
            noise = self.np_rng.normal(0, self.rng.uniform(0, 8), arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            if self.rng.random() < 0.4:
                img = img.filter(ImageFilter.GaussianBlur(self.rng.uniform(0, 1.0)))

        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255.0)
        return t, text


class SynthTextLineDataset(torch.utils.data.Dataset):
    """在线生成式数据集: 每次取数据都随机渲染, 无限量、带无限增强。

    usage: SynthTextLineDataset(alphabet, max_len, num_samples) —— num_samples 只是"伪长度",
    实际每次 __getitem__ 都重新随机生成, 所以 num_samples 可以设很大当"无限"。
    """

    def __init__(self, alphabet, max_len, num_samples=100000, img_h=32, img_w=128,
                 seed=None, font_paths=None, aug_level=2, fixed=False):
        self.alphabet = alphabet
        self.max_len = max_len
        self.gen = SynthTextLineGenerator(img_h=img_h, img_w=img_w, seed=seed,
                                          font_paths=font_paths, aug_level=aug_level)
        self.length = num_samples
        self.fixed = fixed
        if fixed:
            # 预生成 num_samples 张存内存, DataLoader 反复采样 = 有轮次结构
            # (在线模式每张只看一次, 需要百万级步数才收敛; 固定模式几轮就能学)
            imgs, texts = [], []
            for _ in range(num_samples):
                n = random.randint(1, self.max_len)
                text = "".join(random.choices(self.alphabet, k=n))
                img, _ = self.gen.generate(text)
                imgs.append(img)
                texts.append(text)
            self._images = torch.stack(imgs)
            self._texts = texts

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 必须检查越界并抛 IndexError, 否则 `for x in ds` 会无限迭代(不抛错 = 停不下来)
        if idx >= self.length:
            raise IndexError(f"index {idx} out of range {self.length}")
        if self.fixed:
            return self._images[idx], self._texts[idx]
        n = random.randint(1, self.max_len)
        text = "".join(random.choices(self.alphabet, k=n))
        img, _ = self.gen.generate(text)
        return img, text
