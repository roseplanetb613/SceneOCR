"""训练动态模拟: 复现 pretrain_detector.py 的配置, 逐段观察 prob/binary 演化。

对照实验:
  A. 原版配置 (OHEM BCE + thr L1 + binary Dice)
  B. 简化: 纯 masked BCE (无 OHEM, 无 thr 分支, 无 binary)
     —— 隔离出到底是"训练配方"还是"更基础的问题"导致不收敛
"""

import os
import sys
import re
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components.backbone import Hiera
from components.neck import FpnNeck, ImageEncoder, PositionEmbeddingSine
from heads import DBNetHead, db_loss
from data.synthtext import SynthTextDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    model.trunk.load_state_dict(fixed, strict=False)
    print(f"pretrained trunk loaded: {len(fixed)} weights")


def probe(preds, sm, smask, name):
    """打印 prob/binary 在文本区/背景区的统计。"""
    with torch.no_grad():
        prob_p = torch.sigmoid(preds["prob"])
        binary = preds["binary"]
        g = F.interpolate(sm.float(), size=prob_p.shape[-2:], mode="bilinear")
        m = F.interpolate(smask.float(), size=prob_p.shape[-2:], mode="nearest") > 0.5
        txt = g > 0.5
        p_txt, p_bg = prob_p[txt & m].mean().item(), prob_p[(~txt) & m].mean().item()
        b_txt, b_bg = binary[txt & m].mean().item(), binary[(~txt) & m].mean().item()
        print(f"    [{name}] prob_p: all={prob_p.mean().item():.3f} "
              f"txt={p_txt:.3f} bg={p_bg:.3f}  "
              f"binary: all={binary.mean().item():.3f} txt={b_txt:.3f} bg={b_bg:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--mode", choices=["original", "plain"], default="original")
    ap.add_argument("--train_backbone", action="store_true")
    ap.add_argument("--data", choices=["synthtext", "synth"], default="synthtext")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--k", type=float, default=10.0, help="可微分二值化放大系数")
    ap.add_argument("--prob_dice", action="store_true", help="概率图额外加 Dice 损失")
    ap.add_argument("--alpha", type=float, default=1.0, help="概率图损失权重(对照 PaddleOCR 用 5)")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    if args.data == "synthtext":
        ds = SynthTextDataset("E:/迅雷下载/SynthText/SynthText",
                              cache="data/cache_synthtext.pkl",
                              img_size=args.img_size, max_images=20000, seed=1)
    else:
        from data.synth_det import SynthDetDataset
        ds = SynthDetDataset(20000, img_size=args.img_size, seed=1)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    model = build_backbone(DEVICE)
    if not args.no_pretrained:
        load_pretrained_trunk(model, "E:/Sam2/sam2/checkpoints/sam2.1_hiera_tiny.pt")
    for p in model.parameters():
        p.requires_grad_(args.train_backbone)
    head = DBNetHead(in_chans=256, num_levels=3, k=args.k).to(DEVICE)
    head.train()
    params = list(head.parameters()) + (list(model.parameters()) if args.train_backbone else [])
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=0.0, amsgrad=True)

    it = iter(loader)
    print(f"mode={args.mode} steps={args.steps} lr={args.lr} batch={args.batch} "
          f"img={args.img_size} train_backbone={args.train_backbone} "
          f"data={args.data} pretrained={not args.no_pretrained} "
          f"head_params={sum(p.numel() for p in head.parameters()):,} "
          f"total_params={sum(p.numel() for p in params):,}")

    for step in range(args.steps):
        try:
            imgs, sm, smask, tm, tmask = next(it)
        except StopIteration:
            it = iter(loader)
            imgs, sm, smask, tm, tmask = next(it)
        imgs = imgs.to(DEVICE)
        sm, smask, tm, tmask = (sm.to(DEVICE), smask.to(DEVICE),
                                tm.to(DEVICE), tmask.to(DEVICE))

        opt.zero_grad()
        feats = model(imgs)["backbone_fpn"]
        preds = head([f.detach() if not args.train_backbone else f for f in feats])

        if args.mode == "original":
            losses = db_loss(preds, sm, smask, tm, tmask)
            if args.prob_dice:
                prob_p = torch.sigmoid(preds["prob"])
                target = F.interpolate(sm.float(), size=prob_p.shape[-2:], mode="bilinear")
                mask = F.interpolate(smask.float(), size=prob_p.shape[-2:], mode="nearest")
                from heads.losses import dice_masked
                losses["total"] = losses["total"] + dice_masked(prob_p, target, mask)
            if args.alpha != 1.0:
                losses["total"] = (losses["total"]
                                   + (args.alpha - 1.0) * losses["shrink"])
            losses["total"].backward()
            total = losses["total"].item()
            extra = (f"shrink={losses['shrink'].item():.3f} "
                     f"thr={losses['thr'].item():.3f} "
                     f"binary={losses['binary'].item():.3f}")
        else:
            # 纯 masked BCE: 概率图直接监督 shrink_map(全图, 不收缩 OHEM)
            prob_p = torch.sigmoid(preds["prob"])
            loss = F.binary_cross_entropy(
                prob_p.clamp(1e-6, 1 - 1e-6),
                F.interpolate(sm.float(), size=prob_p.shape[-2:], mode="bilinear"))
            loss.backward()
            total = loss.item()
            extra = ""
        opt.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"step {step:4d}: total={total:.4f} {extra}")
            probe(preds, sm, smask, "trn")

    # 最终验证 IoU
    from examples.pretrain_detector import eval_iou
    iou = eval_iou(model, head, ds, DEVICE, num=8)
    print(f"final val-binary-IoU = {iou:.4f}")


if __name__ == "__main__":
    main()
