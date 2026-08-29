"""文本检测头训练 demo：在合成"文本"图上训练 DBNetHead，验证检测组件能学。

做法:
    1. 生成一张合成图(深色底 + 几个白色矩形当"文本块"作 GT);
    2. 冻结骨干(Hiera+FPN)，预计算多尺度特征;
    3. 只训练 DBNetHead，用 DBNet 损失(BCE+Dice+阈值L1);
    4. 看 loss 下降 + 检测框逼近 GT。

注意: 这是单图过拟合冒烟测试, 验证检测头前向/反向/优化器都正常,
不是真实训练(那需要海量文本标注数据)。
"""

import numpy as np
import torch
from PIL import Image, ImageDraw

from components.backbone import Hiera
from components.neck import FpnNeck, ImageEncoder, PositionEmbeddingSine
from heads import DBNetHead, db_loss, prob_to_boxes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256  # 小图, 快速
STEPS = 400
LR = 1e-3


def make_synthetic_text_image(size=256):
    """深色底 + 3 个白色"文本块"，返回 (size,size,3) 图和 GT 掩膜(原分辨率)。"""
    img = Image.new("RGB", (size, size), (30, 30, 30))
    d = ImageDraw.Draw(img)
    boxes = [(20, 30, 180, 55), (40, 80, 220, 105), (15, 130, 160, 155)]  # (x1,y1,x2,y2)
    gt = np.zeros((size, size), dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        d.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
        gt[y1 : y2 + 1, x1 : x2 + 1] = 1.0
    return np.array(img), gt, boxes


def build_backbone():
    trunk = Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2),
                  global_att_blocks=(5, 7, 9), window_pos_embed_bkg_spatial_size=(7, 7))
    neck = FpnNeck(position_encoding=PositionEmbeddingSine(num_pos_feats=256),
                   d_model=256, backbone_channel_list=[768, 384, 192, 96],
                   fpn_top_down_levels=[2, 3], fpn_interp_model="nearest")
    return ImageEncoder(trunk=trunk, neck=neck, scalp=1)


def main():
    torch.manual_seed(0)
    img_np, gt_orig, gt_boxes = make_synthetic_text_image(IMAGE_SIZE)

    # ---- 1. 骨干（冻结）预计算 FPN 特征 ----
    enc = build_backbone().to(DEVICE).eval()
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float().div(255.0)[None].to(DEVICE)
    with torch.no_grad():
        out = enc(img_t)
        features = [f.detach() for f in out["backbone_fpn"]]  # 3 层: 64/32/16
    print("FPN 特征层:", [tuple(f.shape) for f in features])

    # ---- 2. GT 缩放到检测头输出分辨率 (stride-4 = 64x64) ----
    gt = torch.from_numpy(gt_orig)[None, None].to(DEVICE)
    target_hw = features[0].shape[-2:]  # (64, 64)
    gt_small = torch.nn.functional.interpolate(gt, size=target_hw, mode="bilinear", align_corners=False)
    gt_small = (gt_small > 0.5).float()

    # ---- 3. 检测头 + 优化器 ----
    head = DBNetHead(in_chans=256, num_levels=3).to(DEVICE).train()
    opt = torch.optim.AdamW(head.parameters(), lr=LR)
    print(f"检测头参数量: {sum(p.numel() for p in head.parameters()):,}")

    def detect_iou(pred_binary, gt):
        p = pred_binary > 0.5
        g = gt > 0.5
        i = (p & g).sum().float()
        u = (p | g).sum().float()
        return (i / u.clamp(min=1)).item()

    # ---- 4. 训练 ----
    print(f"\n训练 {STEPS} 步 ...")
    for step in range(STEPS):
        opt.zero_grad()
        preds = head(features)
        ones = torch.ones_like(gt_small)
        losses = db_loss(preds, gt_small, ones, gt_small, ones)  # 合成矩形冒烟, 简化 GT
        losses["total"].backward()
        opt.step()
        if step % 80 == 0 or step == STEPS - 1:
            iou = detect_iou(preds["binary"].detach(), gt_small)
            print(f"step {step:3d}: total={losses['total'].item():.4f} "
                  f"(shrink={losses['shrink'].item():.4f}, thr={losses['thr'].item():.4f}, "
                  f"binary={losses['binary'].item():.4f})  binary-IoU={iou:.3f}")

    # ---- 5. 后处理: 概率图 → 检测框 ----
    with torch.no_grad():
        preds = head(features)
        prob = preds["prob"].sigmoid()[0, 0].cpu().numpy()
    boxes = prob_to_boxes(prob, thr=0.3, min_area=16)
    # 64x64 检测框 → 放大回原图 256
    scale = IMAGE_SIZE / target_hw[0]
    boxes_full = [[int(b[0]*scale), int(b[1]*scale), int(b[2]*scale), int(b[3]*scale)] for b in boxes]

    print("\n=== 检测结果 (原图分辨率) ===")
    print(f"GT 框    : {gt_boxes}")
    print(f"检测框   : {boxes_full}")
    overlap = sum(1 for b in boxes_full if any(
        not (b[2] < g[0] or b[0] > g[2] or b[3] < g[1] or b[1] > g[3]) for g in gt_boxes))
    print(f"与 GT 有交集的检测框: {overlap}/{len(gt_boxes)}")


if __name__ == "__main__":
    main()
