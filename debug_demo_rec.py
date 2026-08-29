"""生成识别 demo 网格图: SynthText 真实词裁剪 12 个 (图 + GT + pred), 存 docs/demo/rec_real.png。"""
import os
import sys
import pickle

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from heads import CTCHead
from heads.ctc_utils import ctc_greedy_decode
from heads.recognition_utils import build_vocab

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "docs/demo"
os.makedirs(OUT, exist_ok=True)


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ctc_real62c.pt"
    char_to_idx, idx_to_char = build_vocab(ALPHABET)
    model = CTCHead(d_model=128, vocab_size=len(ALPHABET)).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False)["model"])
    model.eval()

    crops = np.load("data/crops_rec_val.npy")
    with open("data/labels_rec_val.pkl", "rb") as f:
        labels = pickle.load(f)

    # 取前 12 个词
    n = 12
    scale = 4  # 放大 4x
    H, W = 32 * scale, 128 * scale
    gap = 6
    bar_h = 26
    cols, rows = 4, 3
    grid_w = cols * W + (cols + 1) * gap
    grid_h = rows * (H + bar_h) + (rows + 1) * gap
    grid = np.full((grid_h, grid_w, 3), 245, np.uint8)

    with torch.no_grad():
        for i in range(n):
            x = torch.from_numpy(crops[i]).float().div(255.0)[None].to(DEVICE)
            logits = model(x)
            pred = ctc_greedy_decode(logits, idx_to_char, len(ALPHABET))[0]
            gt = labels[i]
            ok = pred == gt

            tile = cv2.resize(crops[i].transpose(1, 2, 0), (W, H),
                              interpolation=cv2.INTER_NEAREST)
            # 边框: 绿=对 红=错
            border = (76, 175, 80) if ok else (229, 57, 53)
            tile = cv2.copyMakeBorder(tile, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=border)

            r, c = divmod(i, cols)
            y0 = gap + r * (H + bar_h + gap)
            x0 = gap + c * (W + gap)
            grid[y0:y0 + H + 6, x0:x0 + W + 6] = tile
            text = f"GT:{gt}  Pred:{pred}  " + ("OK" if ok else "X")
            cv2.putText(grid, text, (x0 + 4, y0 + H + 6 + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 1)

    cv2.imwrite(os.path.join(OUT, "rec_real.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"已保存 docs/demo/rec_real.png ({n} 个真实词)")


if __name__ == "__main__":
    main()
