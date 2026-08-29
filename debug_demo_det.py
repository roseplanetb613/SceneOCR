"""生成检测 demo 图: TD500 真实图 + 合成图 的检测结果可视化, 存 docs/demo/。

每张图: 左=原图, 右=原图+二值图红色叠加, 标题标 IoU。
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
OUT = "docs/demo"
os.makedirs(OUT, exist_ok=True)


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


def visualize(model, head, img, sm=None, fname="demo.png", title=""):
    """img: HWC uint8 RGB (已缩放+pad), sm: GT shrink map (可选)。"""
    x = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)[None].to(DEVICE)
    with torch.no_grad():
        feats = model(x)["backbone_fpn"]
        preds = head([f.detach() for f in feats])
        b = preds["binary"]
        prob = preds["prob"].sigmoid()
    b_np = (b[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
    # 1/4 分辨率 → 原分辨率
    b_full = cv2.resize(b_np * 255, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    vis = img.copy()
    vis[b_full > 127] = (255, 0, 0)  # 红 = 预测文本
    if sm is not None:
        sm_full = cv2.resize((sm > 0.5).astype(np.uint8) * 255,
                             (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        vis_g = img.copy()
        vis_g[sm_full > 127] = (0, 255, 0)  # 绿 = GT
    else:
        vis_g = None

    # 拼图: 原图 | 预测 | (GT)
    panels = [img, vis]
    labels = ["原图", "预测(红)"]
    if vis_g is not None:
        panels.append(vis_g)
        labels.append("GT(绿)")
    grid = np.concatenate(panels, axis=1)
    # 标签条
    bar = np.zeros((30, grid.shape[1], 3), np.uint8)
    for i, lab in enumerate(labels):
        cv2.putText(bar, lab, (i * img.shape[1] + 10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
    grid = np.concatenate([bar, grid], axis=0)
    # 标题
    if title:
        cv2.putText(grid, title, (10, grid.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2)
    cv2.imwrite(os.path.join(OUT, fname), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"已保存 docs/demo/{fname}  {title}")
    return b_np


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/det_td500_neg1b.pt"
    model = build_model(DEVICE)
    head = DBNetHead(in_chans=256, num_levels=3).to(DEVICE).eval()
    ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    head.load_state_dict(ck["head"])
    model.load_state_dict(ck["model"])
    print(f"已加载 {ckpt} (step {ck.get('step','?')})")

    # ---- TD500 test 3 张 ----
    gts = sorted(glob.glob(os.path.join(ROOT, "test", "*.gt")))[:3]
    for k, gt_path in enumerate(gts):
        img_path = gt_path[:-3] + ".JPG"
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = min(IMG / w, IMG / h)
        nh, nw = int(h * scale), int(w * scale)
        ox, oy = (IMG - nw) // 2, (IMG - nh) // 2
        canvas = np.zeros((IMG, IMG, 3), np.uint8)
        canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh))
        polys = parse_td500_gt(gt_path)
        polys = [(p * scale + np.array([ox, oy])).astype(np.float32) for p in polys]
        sm, _, _, _ = make_db_masks(polys, IMG, IMG)
        b = visualize(model, head, canvas, sm, fname=f"det_td500_{k}.png",
                      title=f"TD500 test #{k}")

    # ---- 合成图 demo ----
    from data.synth_det import SynthDetDataset
    ds = SynthDetDataset(1, img_size=IMG, seed=7)
    img_t, sm, _, _, _ = ds[0]
    canvas = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    sm_np = sm[0].numpy()
    visualize(model, head, canvas, sm_np, fname="det_synth_demo.png",
              title="合成数据检测 demo")


if __name__ == "__main__":
    main()
