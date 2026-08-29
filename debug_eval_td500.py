"""确定性评估: 用训好的权重在 TD500 test 上测二值掩膜 IoU (无随机增强)。

流程: 原图 resize 512×512 → 多边形同步缩放 → GT 掩膜 → 模型推理 → 1/4 分辨率 IoU。
"""
import os
import sys
import glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components.backbone import Hiera
from components.neck import FpnNeck, ImageEncoder, PositionEmbeddingSine
from heads import DBNetHead
from data.td500 import parse_td500_gt
from data.db_masks import make_db_masks

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG = 512
ROOT = "C:/Users/Inspiration/Desktop/MSRA-TD500"


def build_model(device):
    trunk = Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2),
                  global_att_blocks=(5, 7, 9), window_pos_embed_bkg_spatial_size=(7, 7))
    neck = FpnNeck(position_encoding=PositionEmbeddingSine(num_pos_feats=256),
                   d_model=256, backbone_channel_list=[768, 384, 192, 96],
                   fpn_top_down_levels=[2, 3], fpn_interp_model="nearest")
    return ImageEncoder(trunk=trunk, neck=neck, scalp=1).to(device).eval()


def binary_iou(pred, gt):
    p, g = pred > 0.5, gt > 0.5
    i, u = (p & g).sum().float(), (p | g).sum().float()
    return (i / u.clamp(min=1)).item()


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/det_td500_neg1b.pt"
    model = build_model(DEVICE)
    head = DBNetHead(in_chans=256, num_levels=3).to(DEVICE).eval()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    head.load_state_dict(ckpt["head"])
    model.load_state_dict(ckpt["model"])
    print(f"已加载 {ckpt_path} (step {ckpt.get('step', '?')}, best_iou {ckpt.get('best_iou', '?')})")

    gts = sorted(glob.glob(os.path.join(ROOT, "test", "*.gt")))
    ious, recalls = [], []
    with torch.no_grad():
        for gt_path in gts:
            img_path = gt_path[:-3] + ".JPG"
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            # 与训练一致的预处理: 等比缩放 + pad 到 512 (居中), 多边形同步缩放
            scale = min(IMG / w, IMG / h)
            nh, nw = int(h * scale), int(w * scale)
            ox, oy = (IMG - nw) // 2, (IMG - nh) // 2
            canvas = np.zeros((IMG, IMG, 3), np.uint8)
            canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh))
            polys = parse_td500_gt(gt_path)
            polys = [(p * scale + np.array([ox, oy])).astype(np.float32) for p in polys]
            sm, smask, tm, tmask = make_db_masks(polys, IMG, IMG)

            img_t = torch.from_numpy(canvas).permute(2, 0, 1).float().div(255.0)[None].to(DEVICE)
            feats = model(img_t)["backbone_fpn"]
            preds = head([f.detach() for f in feats])
            b = preds["binary"]
            g = F.interpolate(torch.from_numpy(sm)[None, None].float().to(DEVICE),
                              size=b.shape[-2:], mode="bilinear")
            ious.append(binary_iou(b, g))
            p = b > 0.5
            gt_bin = g > 0.5
            recalls.append((p & gt_bin).sum().item() / max(gt_bin.sum().item(), 1))

    ious = np.array(ious)
    recalls = np.array(recalls)
    print(f"\nTD500 test 确定性评估 ({len(ious)} 张):")
    print(f"  mean IoU = {ious.mean():.4f}, median IoU = {np.median(ious):.4f}, "
          f"IoU>0.1 占比 = {(ious > 0.1).mean() * 100:.1f}%")
    print(f"  平均召回 = {recalls.mean():.4f} (GT 文本像素被检出比例)")
    print(f"  best IoU = {ious.max():.4f}")


if __name__ == "__main__":
    main()
