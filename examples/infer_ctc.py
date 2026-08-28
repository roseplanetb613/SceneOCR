"""加载训练好的 CTC checkpoint，对合成文本行做单图推理。

用法:
    PYTHONPATH="E:/Sam2/ocr" python examples/infer_ctc.py [--ckpt checkpoints/ctc_full.pt]

渲染测试文本行(和训练同分布) → 模型识别 → 打印 目标 vs 预测。
同时把测试图存到 outputs/ 方便肉眼看。
"""

import argparse
import os

import torch
from PIL import Image

from heads import CTCHead
from heads.ctc_utils import ctc_greedy_decode
from heads.recognition_utils import build_vocab
from data.synth import SynthTextLineGenerator

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_TEXTS = ["HELLO", "OCR2026", "A1B2C3", "SAM2", "XYZ123", "42", "CAT", "9000"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ctc_full.pt")
    ap.add_argument("--img_h", type=int, default=32)
    ap.add_argument("--img_w", type=int, default=128)
    ap.add_argument("--aug_level", type=int, default=2, help="和训练一致的增强")
    args = ap.parse_args()

    # ---- 词表 / 模型 / 权重 ----
    char_to_idx, idx_to_char = build_vocab(ALPHABET)
    blank_idx = len(ALPHABET)  # blank = 词表后第一个索引
    model = CTCHead(d_model=128, vocab_size=len(ALPHABET), img_h=args.img_h).to(DEVICE)
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"加载权重: {args.ckpt} (训练到 step {ckpt['step']})")

    # ---- 渲染测试图 + 推理 ----
    gen = SynthTextLineGenerator(args.img_h, args.img_w, seed=0, aug_level=args.aug_level)
    os.makedirs("outputs", exist_ok=True)
    ok = 0
    print(f"\n{'目标':<10}{'预测':<10}  结果")
    print("-" * 30)
    for i, text in enumerate(TEST_TEXTS):
        img, _ = gen.generate(text)
        # 存图供肉眼看
        pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype("uint8"))
        pil.save(f"outputs/infer_{i}_{text}.png")
        with torch.no_grad():
            logits = model(img.unsqueeze(0).to(DEVICE))
        pred = ctc_greedy_decode(logits, idx_to_char, blank_idx)[0]
        mark = "[OK]" if pred == text else "[X]"
        ok += pred == text
        print(f"{text:<10}{pred:<10} {mark}  (图: outputs/infer_{i}_{text}.png)")

    print(f"\n识别 {ok}/{len(TEST_TEXTS)} 正确")


if __name__ == "__main__":
    main()
