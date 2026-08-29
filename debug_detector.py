"""诊断 DBNet 检测头为什么学不到东西。

检查项:
  1. 骨干特征是否退化(常量/零/数值爆炸) —— 决定检测头有没有"原料"
  2. GT 掩膜是否与图像对齐 —— 决定监督信号对不对
  3. 检测头输出统计(prob/thr/binary) —— 判断输出是否塌缩
  4. 梯度流向 —— 哪个模块拿不到梯度
"""

import os
import sys
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components.backbone import Hiera
from components.neck import FpnNeck, ImageEncoder, PositionEmbeddingSine
from heads import DBNetHead, db_loss
from data.synthtext import SynthTextDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)


def build_backbone(device):
    trunk = Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2),
                  global_att_blocks=(5, 7, 9), window_pos_embed_bkg_spatial_size=(7, 7))
    neck = FpnNeck(position_encoding=PositionEmbeddingSine(num_pos_feats=256),
                   d_model=256, backbone_channel_list=[768, 384, 192, 96],
                   fpn_top_down_levels=[2, 3], fpn_interp_model="nearest")
    return ImageEncoder(trunk=trunk, neck=neck, scalp=1).to(device).eval()


def load_pretrained_trunk(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
    prefix = "image_encoder.trunk."
    trunk_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    fixed = {}
    for k, v in trunk_sd.items():
        if re.search(r"\.mlp\.layers\.\d+\.(weight|bias)$", k):
            k = re.sub(r"(\.mlp\.layers\.\d+)\.(weight|bias)$", r"\1.fc.\2", k)
        fixed[k] = v
    missing, unexpected = model.trunk.load_state_dict(fixed, strict=False)
    print(f"预训练骨干加载: {len(fixed)} 个权重, missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("  missing 示例:", list(missing)[:5])
    if unexpected:
        print("  unexpected 示例:", list(unexpected)[:5])
    return missing, unexpected


def tensor_stats(name, t):
    t = t.detach().float()
    print(f"  {name}: shape={tuple(t.shape)} mean={t.mean().item():.4f} "
          f"std={t.std().item():.4f} min={t.min().item():.4f} max={t.max().item():.4f} "
          f"spatial_std={t.std(dim=(-2, -1)).mean().item():.4f}")


def main():
    # ---------- 1. 数据样本 + GT 对齐检查 ----------
    ds = SynthTextDataset("E:/迅雷下载/SynthText/SynthText",
                          cache="data/cache_synthtext.pkl",
                          img_size=640, max_images=20000, seed=1)
    print(f"\n===== 1. SynthText 样本检查 (共 {len(ds)} 张) =====")
    img, sm, smask, tm, tmask = ds[0]
    print("img:", img.shape, "shrink_map 前景占比:",
          (sm > 0.5).float().mean().item() * 100, "%")
    print("shrink_mask 有效区域占比:", (smask > 0.5).float().mean().item() * 100, "%")
    print("threshold_map 非零占比:", (tm > 0).float().mean().item() * 100, "%")

    # 画 overlay 检查对齐
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    sm_np = (sm[0].numpy() * 255).astype(np.uint8)
    tm_np = (tm[0].numpy() * 255).astype(np.uint8)
    os.makedirs("outputs/debug", exist_ok=True)
    cv2.imwrite("outputs/debug/gt_align_img.png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite("outputs/debug/gt_shrink.png", sm_np)
    cv2.imwrite("outputs/debug/gt_threshold.png", tm_np)
    # 红色叠加 shrink map 在图上
    overlay = img_np.copy()
    overlay[sm_np > 127] = (255, 0, 0)
    cv2.imwrite("outputs/debug/gt_align_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("已保存 GT 对齐可视化: outputs/debug/gt_align_*.png")

    # ---------- 2. 骨干 + 检测头 ----------
    print("\n===== 2. 骨干特征检查 =====")
    model = build_backbone(DEVICE)
    load_pretrained_trunk(model, "E:/Sam2/sam2/checkpoints/sam2.1_hiera_tiny.pt")
    head = DBNetHead(in_chans=256, num_levels=3).to(DEVICE)

    # 用一批真实样本检查特征
    imgs = torch.stack([ds[i][0] for i in range(4)]).to(DEVICE)
    with torch.no_grad():
        feats = model(imgs)["backbone_fpn"]
    for i, f in enumerate(feats):
        tensor_stats(f"FPN level {i}", f)

    # ---------- 3. 检测头输出 ----------
    print("\n===== 3. 检测头输出统计 (训练模式) =====")
    head.train()
    with torch.no_grad():
        preds = head([f.clone() for f in feats])
    for name in ["prob", "thr", "binary"]:
        tensor_stats(name, preds[name])
    prob_p = torch.sigmoid(preds["prob"])
    print(f"  prob_p (sigmoid): mean={prob_p.mean().item():.4f} "
          f"pos_frac(>0.5)={ (prob_p > 0.5).float().mean().item()*100:.2f}%")
    print(f"  binary > 0.5 占比: {(preds['binary'] > 0.5).float().mean().item()*100:.4f}%")

    # ---------- 4. 梯度检查 ----------
    print("\n===== 4. 梯度检查 (1 步 forward/backward) =====")
    sm_b, smask_b, tm_b, tmask_b = (torch.stack([ds[i][j] for i in range(4)])
                                     for j in [1, 2, 3, 4])
    sm_b, smask_b, tm_b, tmask_b = (sm_b.to(DEVICE), smask_b.to(DEVICE),
                                    tm_b.to(DEVICE), tmask_b.to(DEVICE))
    feats_req = model(imgs)["backbone_fpn"]
    preds = head([f.detach() for f in feats_req])
    losses = db_loss(preds, sm_b, smask_b, tm_b, tmask_b)
    losses["total"].backward()
    print(f"  loss: total={losses['total'].item():.4f} "
          f"shrink={losses['shrink'].item():.4f} "
          f"thr={losses['thr'].item():.4f} "
          f"binary={losses['binary'].item():.4f}")
    for name, p in head.named_parameters():
        if p.grad is not None:
            print(f"  head.{name}: grad_norm={p.grad.norm().item():.6f} "
                  f"param_norm={p.detach().norm().item():.4f}")
        else:
            print(f"  head.{name}: NO GRAD")

    # ---------- 5. 输出图与 GT 对比(数值) ----------
    print("\n===== 5. prob/binary 与 GT 的区域一致性 =====")
    with torch.no_grad():
        prob_p2 = torch.sigmoid(preds["prob"])  # (B,1,H,W)
        binary2 = preds["binary"]
        g = F.interpolate(sm_b.float(), size=prob_p2.shape[-2:], mode="bilinear")
        for name, pred in [("prob_p", prob_p2), ("binary", binary2)]:
            pred_txt = pred[g > 0.5].mean().item()
            pred_bg = pred[g <= 0.5].mean().item()
            print(f"  {name}: 文本区内均值={pred_txt:.4f}, 背景区均值={pred_bg:.4f} "
                  f"(差异 {pred_txt - pred_bg:+.4f})")


if __name__ == "__main__":
    main()
