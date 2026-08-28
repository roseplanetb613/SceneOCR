"""DBNet 文本检测训练：在 MSRA-TD500 真实数据上训练检测头。

用法:
    PYTHONPATH="E:/Sam2/ocr" python examples/pretrain_detector.py \
        --data C:/Users/Inspiration/Desktop/MSRA-TD500 \
        --img_size 512 --batch 2 --steps 1000 \
        --ckpt checkpoints/det_td500.pt --log logs/det_td500.log
    # 续训:
    ... python examples/pretrain_detector.py --resume ...

默认冻结骨干(只训 DBNetHead, 快且稳); 加 --train_backbone 才一起训练骨干。
"""

import argparse
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from components.backbone import Hiera
from components.neck import FpnNeck, ImageEncoder, PositionEmbeddingSine
from heads import DBNetHead, db_loss
from data.td500 import MSRATD500Dataset
from data.synth_det import SynthDetDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== checkpoint / 日志 ====================
def save_checkpoint(path, model, head, opt, sched, step, best_iou, args):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {"head": head.state_dict(), "model": model.state_dict(),
         "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
         "step": step, "best_iou": best_iou, "args": vars(args)}, path)


def load_checkpoint(path, model, head, opt, sched, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    head.load_state_dict(ckpt["head"])
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    sched.load_state_dict(ckpt["scheduler"])
    return ckpt["step"], ckpt["best_iou"]


def log_line(path, text, echo=True):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    if echo:
        print(text)


# ==================== 骨干 ====================
def build_backbone(device):
    trunk = Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2),
                  global_att_blocks=(5, 7, 9), window_pos_embed_bkg_spatial_size=(7, 7))
    neck = FpnNeck(position_encoding=PositionEmbeddingSine(num_pos_feats=256),
                   d_model=256, backbone_channel_list=[768, 384, 192, 96],
                   fpn_top_down_levels=[2, 3], fpn_interp_model="nearest")
    return ImageEncoder(trunk=trunk, neck=neck, scalp=1).to(device).eval()


# ==================== 评估 ====================
def binary_iou(pred_binary, gt):
    p = pred_binary > 0.5
    g = gt > 0.5
    i = (p & g).sum().float()
    u = (p | g).sum().float()
    return (i / u.clamp(min=1)).item()


def eval_iou(model, head, ds, device, num=8):
    model.eval()
    head.eval()
    ious = []
    with torch.no_grad():
        for i in range(min(num, len(ds))):
            img, full, prob, thr = ds[i]
            img = img[None].to(device)
            feats = model(img)["backbone_fpn"]
            preds = head([f.detach() for f in feats])
            b = preds["binary"]
            # GT 补 batch 维 + 缩放到检测头输出分辨率再比 IoU
            g = F.interpolate(full[None].to(device).float(), size=b.shape[-2:], mode="bilinear")
            ious.append(binary_iou(b, (g > 0.5).float()))
    return sum(ious) / max(len(ious), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="C:/Users/Inspiration/Desktop/MSRA-TD500",
                    help="真实数据目录, 或 'synth' 用合成检测数据预训练")
    ap.add_argument("--synth_samples", type=int, default=10000, help="--data synth 时的样本量")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train_backbone", action="store_true", help="一起训练骨干(慢)")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--ckpt", default="checkpoints/det_td500.pt")
    ap.add_argument("--log", default="logs/det_td500.log")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init_ckpt", default=None,
                    help="只加载 head+backbone 权重(重置优化器/步数), 用于低 lr 微调")
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)

    if args.data == "synth":
        train_ds = SynthDetDataset(args.synth_samples, img_size=args.img_size, seed=1)
        val_ds = SynthDetDataset(50, img_size=args.img_size, seed=999)
        log_line(args.log, f"== 合成检测预训练 (train {args.synth_samples}, "
                 f"img {args.img_size}, backbone训练={args.train_backbone}) ==")
    else:
        train_ds = MSRATD500Dataset(args.data, split="train", img_size=args.img_size)
        val_ds = MSRATD500Dataset(args.data, split="test", img_size=args.img_size)
        log_line(args.log, f"== 检测微调 (train {len(train_ds)}, test {len(val_ds)}, "
                 f"img {args.img_size}, backbone训练={args.train_backbone}) ==")

    model = build_backbone(DEVICE)
    for p in model.parameters():
        p.requires_grad_(args.train_backbone)  # 默认冻结骨干
    head = DBNetHead(in_chans=256, num_levels=3).to(DEVICE)
    params = list(head.parameters()) + (list(model.parameters()) if args.train_backbone else [])
    opt = torch.optim.AdamW(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    print(f"DBNetHead 参数: {sum(p.numel() for p in head.parameters()):,}")

    start_step, best_iou = 0, 0.0
    if args.resume and os.path.exists(args.ckpt):
        start_step, best_iou = load_checkpoint(args.ckpt, model, head, opt, sched, DEVICE)
        log_line(args.log, f"== 续训 (step {start_step}) ==")
    elif args.init_ckpt and os.path.exists(args.init_ckpt):
        ckpt = torch.load(args.init_ckpt, map_location=DEVICE, weights_only=False)
        head.load_state_dict(ckpt["head"])
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        log_line(args.log, f"== 从 {args.init_ckpt} 载入权重(优化器重置, 从头微调) ==")

    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    it = iter(loader)
    t0 = time.time()

    try:
        for step in range(start_step, args.steps):
            try:
                imgs, full, prob, thr = next(it)
            except StopIteration:
                it = iter(loader)
                imgs, full, prob, thr = next(it)
            imgs = imgs.to(DEVICE)
            full, prob, thr = full.to(DEVICE), prob.to(DEVICE), thr.to(DEVICE)
            opt.zero_grad()
            with torch.no_grad():
                feats = model(imgs)["backbone_fpn"]
            preds = head([f.detach() for f in feats])
            losses = db_loss(preds, full, prob_gt=prob, thr_gt=thr)
            losses["total"].backward()
            opt.step()
            sched.step()

            if step % args.log_every == 0 or step == args.steps - 1:
                iou = eval_iou(model, head, val_ds, DEVICE, num=8)
                spd = (step + 1 - start_step) / max(time.time() - t0, 1e-6)
                log_line(args.log, f"step {step:5d}: total={losses['total'].item():.4f} "
                         f"(prob_bce={losses['prob_bce'].item():.3f}, dice={losses['dice'].item():.3f}) "
                         f"val-binary-IoU={iou:.3f}  ({spd:.1f} 步/s)")
                if iou > best_iou:
                    best_iou = iou
                    save_checkpoint(args.ckpt, model, head, opt, sched, step + 1, best_iou, args)
                    log_line(args.log, f"  → 新 best IoU={iou:.3f}")

            if (step + 1) % args.save_every == 0:
                save_checkpoint(args.ckpt, model, head, opt, sched, step + 1, best_iou, args)
                log_line(args.log, f"  → 断点已存 step {step+1}")

    except KeyboardInterrupt:
        save_checkpoint(args.ckpt, model, head, opt, sched, step + 1, best_iou, args)
        log_line(args.log, f"== 被中断, 断点已存 step {step+1} ==")
        return

    log_line(args.log, f"== 完成. best val-binary-IoU: {best_iou:.3f}, checkpoint: {args.ckpt} ==")


if __name__ == "__main__":
    main()
