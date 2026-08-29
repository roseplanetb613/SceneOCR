"""端到端 OCR（全自研版）：自研 DBNet 检测 + 自研 CTC 识别。

对比 examples/ocr_end2end.py（CRAFT 检测版）——本脚本检测也用自研 DBNetHead：
    真实图片 → [自研 DBNet 检测] → 文本框 → 裁剪 → [自研 CTC 识别] → 文字

用法:
    PYTHONPATH="E:/Sam2/ocr" python examples/ocr_end2end_dbnet.py \
        --image <图> --det_ckpt checkpoints/det_synth_v2.pt \
        --rec_ckpt checkpoints/ctc_real62c.pt

域匹配建议: 合成图用 det_synth_v2.pt, 真实图用 det_td500_neg1b.pt。
注意: 自研检测头目前只在 300 张 TD500 上微调(确定性 mean IoU ~0.06), 训练域外的
真实图基本不可用——本脚本的意义是展示"全自研链路"已打通, 检测精度需长训练提升
(README 已诚实标注)。识别部分(ctc_real62c.pt)在真实词裁剪上 60.6% 可用。
"""

import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image

from heads import DBNetHead, prob_to_boxes
from heads import CTCHead
from heads.ctc_utils import ctc_greedy_decode
from heads.recognition_utils import build_vocab
from components.backbone import Hiera
from components.neck import FpnNeck, ImageEncoder, PositionEmbeddingSine

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DET_IMG = 512  # 检测输入尺寸


def build_detector(det_ckpt):
    """加载自研 DBNet 检测器 (冻结骨干 + DBNetHead)。"""
    trunk = Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2),
                  global_att_blocks=(5, 7, 9), window_pos_embed_bkg_spatial_size=(7, 7))
    neck = FpnNeck(position_encoding=PositionEmbeddingSine(num_pos_feats=256),
                   d_model=256, backbone_channel_list=[768, 384, 192, 96],
                   fpn_top_down_levels=[2, 3], fpn_interp_model="nearest")
    enc = ImageEncoder(trunk=trunk, neck=neck, scalp=1).to(DEVICE).eval()
    head = DBNetHead(in_chans=256, num_levels=3).to(DEVICE).eval()
    ck = torch.load(det_ckpt, map_location=DEVICE, weights_only=False)
    head.load_state_dict(ck["head"])
    enc.load_state_dict(ck["model"])
    print(f"已加载自研检测器: {det_ckpt}")
    return enc, head


def load_recognizer(ckpt_path):
    char_to_idx, idx_to_char = build_vocab(ALPHABET)
    model = CTCHead(d_model=128, vocab_size=len(ALPHABET)).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False)["model"])
    model.eval()
    print(f"已加载自研识别器: {ckpt_path}")
    return model, idx_to_char


def detect_boxes(enc, head, img_bgr):
    """整图检测 → 文本行框列表 [(x1,y1,x2,y2), ...] (原图坐标)。"""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = min(DET_IMG / w, DET_IMG / h)
    nh, nw = int(h * scale), int(w * scale)
    ox, oy = (DET_IMG - nw) // 2, (DET_IMG - nh) // 2
    canvas = np.zeros((DET_IMG, DET_IMG, 3), np.uint8)
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh))
    x = torch.from_numpy(canvas).permute(2, 0, 1).float().div(255.0)[None].to(DEVICE)
    with torch.no_grad():
        feats = enc(x)["backbone_fpn"]
        preds = head([f.detach() for f in feats])
        prob = preds["prob"].sigmoid()[0, 0].cpu().numpy()  # 1/4 分辨率
    # 概率图 → 框 (1/4 分辨率), 再映射回原图
    boxes_small = prob_to_boxes(prob, thr=0.4, min_area=8)
    det_scale = DET_IMG / prob.shape[0]  # 1/4 分辨率 → 检测输入
    boxes = []
    for x1, y1, x2, y2 in boxes_small:
        bx1 = int((x1 * det_scale - ox) / scale)
        by1 = int((y1 * det_scale - oy) / scale)
        bx2 = int((x2 * det_scale - ox) / scale)
        by2 = int((y2 * det_scale - oy) / scale)
        # 防御: 裁剪到图内 + 过滤无效框
        bx1, by1 = max(bx1, 0), max(by1, 0)
        bx2, by2 = min(bx2, w - 1), min(by2, h - 1)
        if bx2 - bx1 < 4 or by2 - by1 < 4:
            continue
        boxes.append((bx1, by1, bx2, by2))
    return boxes, prob


def recognize_crop(model, idx_to_char, crop):
    img = cv2.resize(crop, (128, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)[None].to(DEVICE)
    with torch.no_grad():
        logits = model(x)
    return ctc_greedy_decode(logits, idx_to_char, len(ALPHABET))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="E:/Sam2/sam2/test.jpg")
    ap.add_argument("--det_ckpt", default="checkpoints/det_td500_neg1b.pt")
    ap.add_argument("--rec_ckpt", default="checkpoints/ctc_real62c.pt")
    ap.add_argument("--out", default="outputs/ocr_end2end_dbnet.png")
    args = ap.parse_args()

    enc, head = build_detector(args.det_ckpt)
    model, idx_to_char = load_recognizer(args.rec_ckpt)

    img = cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR)
    boxes, prob = detect_boxes(enc, head, img)
    print(f"\n自研检测到 {len(boxes)} 个文本框\n")
    print(f"{'检测框':<28}{'识别结果':<15}")
    print("-" * 45)
    results = []
    for x1, y1, x2, y2 in boxes:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        text = recognize_crop(model, idx_to_char, crop)
        results.append((x1, y1, x2, y2, text))
        print(f"[{x1},{y1} {x2},{y2}]".ljust(28) + f"{text:<15}")

    vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for x1, y1, x2, y2, text in results:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(vis, text, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    Image.fromarray(vis).save(args.out)
    print(f"\n可视化已保存: {args.out}")


if __name__ == "__main__":
    main()
