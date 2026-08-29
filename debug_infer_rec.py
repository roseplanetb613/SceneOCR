"""推理检查: 用 ctc_real62b.pt 识别 SynthText 验证集的真实词裁剪, 打印对错样例。"""
import os
import sys
import pickle

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from heads import CTCHead
from heads.ctc_utils import ctc_greedy_decode
from heads.recognition_utils import build_vocab

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ctc_real62b.pt"
    char_to_idx, idx_to_char = build_vocab(ALPHABET)
    model = CTCHead(d_model=128, vocab_size=len(ALPHABET)).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False)["model"])
    model.eval()

    crops = np.load("data/crops_rec_val.npy")
    with open("data/labels_rec_val.pkl", "rb") as f:
        labels = pickle.load(f)

    n = len(labels)
    hits, chars_hit, chars_total = 0, 0, 0
    samples = []
    with torch.no_grad():
        for i in range(n):
            x = torch.from_numpy(crops[i]).float().div(255.0)[None].to(DEVICE)
            logits = model(x)
            pred = ctc_greedy_decode(logits, idx_to_char, len(ALPHABET))[0]
            gt = labels[i]
            if pred == gt:
                hits += 1
            else:
                samples.append((i, gt, pred))
            chars_total += max(len(gt), 1)
            for a, b in zip(pred, gt):
                if a == b:
                    chars_hit += 1
    print(f"模型: {ckpt}")
    print(f"整串准确率: {hits/n:.3f} ({hits}/{n})  字符准确率: {chars_hit/chars_total:.3f}")
    print("\n前 25 个错误样例 (idx, GT, pred):")
    for s in samples[:25]:
        print(f"  {s}")

    # 对错各展示 5 个(文本)
    print("\n正确样例 (前 10):")
    shown = 0
    with torch.no_grad():
        for i in range(n):
            x = torch.from_numpy(crops[i]).float().div(255.0)[None].to(DEVICE)
            pred = ctc_greedy_decode(model(x), idx_to_char, len(ALPHABET))[0]
            if pred == labels[i]:
                print(f"  [{i}] GT={labels[i]!r} → {pred!r}")
                shown += 1
                if shown >= 10:
                    break


if __name__ == "__main__":
    main()
