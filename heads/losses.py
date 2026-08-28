"""DBNet 训练损失：BCE(二值图) + Dice(概率图) + L1(阈值图)。

原理:
    - 二值图损失: 让二值图 B 逼近 GT 掩膜（文本区域=1）;
    - 概率图损失: Dice，对前景/背景不平衡更鲁棒（文本区域通常占比小）;
    - 阈值图损失: 让阈值图在文本边界带=1（用膨胀-腐蚀构造的边界 GT 监督），
      其余=0。阈值图学得越准，二值化的 k*(P-T) 分界线越贴文本边缘。
"""

import torch
import torch.nn.functional as F


def dice_loss(inputs, targets, eps=1.0):
    """Dice 损失 ≈ 1 - IoU，对掩膜分割常用。"""
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1)
    return (1 - (numerator + eps) / (denominator + eps)).mean()


def ohem_bce(logits, targets, neg_pos_ratio=3):
    """在线难例挖掘 BCE：取全部正样本 + 最难的 top-k 负样本。

    文本像素占比极小(1~3%), 普通 BCE 会被背景主导 → 模型学成"全背景"。
    OHEM 只回传: 所有文本像素 + 损失最大的负样本(难例), 逼模型关注前景。
    """
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss_flat, tgt_flat = loss.flatten(1), targets.flatten(1)
    ohem = []
    for b in range(logits.shape[0]):
        pos = tgt_flat[b] > 0.5
        num_pos = int(pos.sum().item())
        if num_pos == 0:
            continue  # 这张图没有前景(如 hard 全过滤), 跳过避免空 mean → NaN
        neg = ~pos
        num_neg = min(int(num_pos * neg_pos_ratio), int(neg.sum().item()))
        if num_neg > 0:
            topk, _ = torch.topk(loss_flat[b][neg], num_neg)
            sel = torch.cat([loss_flat[b][pos], topk])
        else:
            sel = loss_flat[b][pos]
        ohem.append(sel.mean())
    if ohem:
        return torch.stack(ohem).mean()
    return loss_flat.new_tensor(0.0)  # 整批都无前景 → 0 损失


def make_boundary_map(gt, kernel=3):
    """用膨胀 - 腐蚀构造"边界带"GT：文本边界一圈 = 1，其余 = 0。

    纯 torch 实现，不需要 cv2：
        膨胀 = maxpool(gt)（往外扩一圈）
        腐蚀 = -maxpool(-gt)（往里缩一圈）
        边界带 = 膨胀 - 腐蚀（只在边缘一圈是 1）
    """
    pad = kernel // 2
    dilate = F.max_pool2d(gt.float(), kernel, stride=1, padding=pad)
    erode = -F.max_pool2d(-gt.float(), kernel, stride=1, padding=pad)
    return (dilate - erode).clamp(0, 1)


def db_loss(
    preds: dict,          # DBNetHead 输出: {prob, thr, binary}
    gt: torch.Tensor,     # GT 完整文本掩膜 (B,1,H,W)，监督二值图
    prob_gt: torch.Tensor = None,  # 收缩后的概率图 GT（DBNet 用收缩掩膜监督 prob）
    thr_gt: torch.Tensor = None,   # 边界带阈值图 GT（DBNet 用边界带监督 thr）
    alpha: float = 10.0,  # 阈值图损失权重 (DBNet 原文 α=10)
    beta: float = 1.0,    # 二值图损失权重 (DBNet 原文 β=1)
    thr_kernel: int = 5,  # 无 thr_gt 时自动构造边界带的核大小
):
    """DBNet 损失 = L_s(概率图) + α L_t(阈值图) + β L_b(二值图)。

    关键: L_s 对 prob 图直接算 BCE+Dice(不经过饱和的 DB binary),
    保证梯度始终能流向概率头 —— 否则 k=50 的 DB 一饱和梯度就消失、训练卡死。
    """
    prob, thr, binary = preds["prob"], preds["thr"], preds["binary"]

    def _resize_bin(m):
        """缩放到输出分辨率并二值化。"""
        target_hw = prob.shape[-2:]
        if m.shape[-2:] != target_hw:
            m = F.interpolate(m.float(), size=target_hw, mode="bilinear", align_corners=False)
        return (m > 0.5).float()

    gt_b = _resize_bin(gt)                     # 完整掩膜 → 二值图监督
    prob_gt_b = _resize_bin(prob_gt if prob_gt is not None else gt)
    if thr_gt is None:
        thr_gt = make_boundary_map(gt_b, kernel=thr_kernel)
    thr_gt_b = _resize_bin(thr_gt)

    # L_s: 概率图 (OHEM BCE + Dice) 对收缩 GT —— 主要的梯度来源
    prob_bce = ohem_bce(prob, prob_gt_b)  # 难例挖掘, 避免被背景主导
    prob_dice = dice_loss(prob.sigmoid(), prob_gt_b)
    loss_s = prob_bce + prob_dice

    # L_t: 阈值图 L1 对边界带
    loss_t = F.l1_loss(thr.sigmoid(), thr_gt_b, reduction="mean")

    # L_b: 二值图 BCE 对完整 GT（通过 DB, 饱和时梯度弱, 辅助作用）
    loss_b = F.binary_cross_entropy(binary, gt_b, reduction="mean")

    total = loss_s + alpha * loss_t + beta * loss_b
    return {
        "total": total,
        "prob_bce": prob_bce,
        "dice": prob_dice,
        "thr": loss_t,
        "binary_bce": loss_b,
    }
