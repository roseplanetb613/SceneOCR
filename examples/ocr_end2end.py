"""端到端 OCR：成熟检测器(CRAFT, easyocr) + 自研 CTC 识别器。

检测用成熟预训练权重(easyocr 的 CRAFT), 识别用我们训练的 CTCHead(ctc_full.pt, 36字符 94.5%)。
这样检测立即可用(不用训), 识别是自研的 —— 端到端能读真实图片文字。
"""

import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from heads import CTCHead
from heads.ctc_utils import ctc_greedy_decode
from heads.recognition_utils import build_vocab

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_recognizer(ckpt_path):
    """加载自研 CTC 识别器。"""
    char_to_idx, idx_to_char = build_vocab(ALPHABET)
    model = CTCHead(d_model=128, vocab_size=len(ALPHABET)).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False)["model"])
    model.eval()
    return model, idx_to_char


def recognize_crop(model, idx_to_char, crop):
    """识别一个文本行裁剪 (H,W,3) → 字符串。"""
    img = cv2.resize(crop, (128, 32))  # 缩放到识别器输入
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)[None].to(DEVICE)
    with torch.no_grad():
        logits = model(x)
    blank = len(ALPHABET)
    return ctc_greedy_decode(logits, idx_to_char, blank)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="E:/Sam2/sam2/test.jpg")
    ap.add_argument("--ckpt", default="checkpoints/ctc_real62c.pt",
                    help="识别权重 (62字符词表: ctc_synth62 / ctc_real62b / ctc_real62c)")
    ap.add_argument("--out", default="outputs/ocr_end2end.png")
    args = ap.parse_args()

    import easyocr
    print("加载 CRAFT 检测器(easyocr)...")
    reader = easyocr.Reader(["en"], gpu=DEVICE.startswith("cuda"))
    print("加载自研 CTC 识别器...")
    model, idx_to_char = load_recognizer(args.ckpt)

    img = cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR)
    hboxes, fboxes = reader.detect(img)
    print(f"\n检测到 {len(hboxes[0])} 个文本框\n")
    print(f"{'检测框':<28}{'识别结果':<15}")
    print("-" * 45)

    results = []
    for box in hboxes[0]:
        x1, x2, y1, y2 = [int(v) for v in box]
        crop = img[y1:y2, x1:x2]
        text = recognize_crop(model, idx_to_char, crop)
        results.append((x1, y1, x2, y2, text))
        print(f"[{x1},{y1} {x2},{y2}]".ljust(28) + f"{text:<15}")

    # 可视化: 画框 + 标注识别结果
    vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for x1, y1, x2, y2, text in results:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
    Image.fromarray(vis).save(args.out)
    print(f"\n可视化已保存: {args.out}")


if __name__ == "__main__":
    main()
