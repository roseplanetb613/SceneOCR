"""加载训练好的 CTC checkpoint 做识别推理 demo。

默认: 从 SynthText 真实词裁剪验证集(val 预裁剪缓存)抽样本识别, 展示真实能力。
可选: --synth_demo 渲染合成文本行测试(注意: 真实微调模型对合成渲染分布有领域遗忘,
      合成 demo 分数不代表真实场景能力, 仅作流程演示)。

用法:
    PYTHONPATH="E:/Sam2/ocr" python examples/infer_ctc.py [--ckpt checkpoints/ctc_real62c.pt]
    PYTHONPATH="E:/Sam2/ocr" python examples/infer_ctc.py --synth_demo
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from heads import CTCHead
from heads.ctc_utils import ctc_greedy_decode
from heads.recognition_utils import build_vocab
from data.synth import SynthTextLineGenerator

# 62 字符: 数字 + 大写 + 小写 (与识别训练一致)
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYNTH_TEXTS = ["Hello", "OCR2026", "A1b2C3", "Sam2", "xyz123", "42", "Cat", "9000"]


def load_model(ckpt_path, img_h):
    char_to_idx, idx_to_char = build_vocab(ALPHABET)
    model = CTCHead(d_model=128, vocab_size=len(ALPHABET), img_h=img_h).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE,
                                     weights_only=False)["model"])
    model.eval()
    return model, idx_to_char


def recognize(model, idx_to_char, x):
    with torch.no_grad():
        logits = model(x)
    return ctc_greedy_decode(logits, idx_to_char, len(ALPHABET))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ctc_real62c.pt")
    ap.add_argument("--img_h", type=int, default=32)
    ap.add_argument("--img_w", type=int, default=128)
    ap.add_argument("--synth_demo", action="store_true",
                    help="用合成渲染测试文本行(默认: SynthText 真实词裁剪验证集)")
    ap.add_argument("--n_show", type=int, default=20, help="真实裁剪展示条数")
    args = ap.parse_args()

    model, idx_to_char = load_model(args.ckpt, args.img_h)
    print(f"加载权重: {args.ckpt}")

    if args.synth_demo:
        # ---- 合成渲染 demo ----
        gen = SynthTextLineGenerator(args.img_h, args.img_w, seed=0, aug_level=2)
        os.makedirs("outputs", exist_ok=True)
        print(f"\n{'目标':<10}{'预测':<10}  结果")
        print("-" * 30)
        ok = 0
        for i, text in enumerate(SYNTH_TEXTS):
            img, _ = gen.generate(text)
            pred = recognize(model, idx_to_char, img.unsqueeze(0).to(DEVICE))
            pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype("uint8"))
            pil.save(f"outputs/infer_{i}_{text}.png")
            mark = "[OK]" if pred == text else "[X]"
            ok += pred == text
            print(f"{text:<10}{pred:<10} {mark}  (图: outputs/infer_{i}_{text}.png)")
        print(f"\n合成 demo 识别 {ok}/{len(SYNTH_TEXTS)} 正确")
        print("提示: 真实微调模型对合成渲染分布有领域遗忘, 该分数不代表真实场景能力。")
        return

    # ---- 真实词裁剪验证集 demo ----
    crops = np.load("data/crops_rec_val.npy")
    with open("data/labels_rec_val.pkl", "rb") as f:
        import pickle
        labels = pickle.load(f)
    os.makedirs("outputs", exist_ok=True)
    n = min(args.n_show, len(labels))
    hits = 0
    print(f"\nSynthText 真实词裁剪验证集 (前 {n} 个):")
    print(f"{'GT':<12}{'预测':<12}  结果")
    print("-" * 32)
    for i in range(n):
        x = torch.from_numpy(crops[i]).float().div(255.0)[None].to(DEVICE)
        pred = recognize(model, idx_to_char, x)
        gt = labels[i]
        mark = "[OK]" if pred == gt else "[X]"
        hits += pred == gt
        Image.fromarray(crops[i].transpose(1, 2, 0)).save(f"outputs/rec_infer_{i}_{gt}.png")
        print(f"{gt:<12}{pred:<12} {mark}")
    print(f"\n整串准确率: {hits}/{n} ({hits/n:.1%})")


if __name__ == "__main__":
    main()
