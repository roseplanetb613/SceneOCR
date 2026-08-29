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


def ohem_bce(logits_or_prob, targets, neg_pos_ratio=3):
    """在线难例挖掘 BCE：取全部正样本 + 最难的 top-k 负样本。

    文本像素占比极小(1~3%), 普通 BCE 会被背景主导 → 模型学成"全背景"。
    OHEM 只回传: 所有文本像素 + 损失最大的负样本(难例), 逼模型关注前景。
    自动识别输入是 logits 还是概率(0~1): 概率范围明显在 (-1,1.5) 内视为 sigmoid 概率。
    """
    x = logits_or_prob
    if x.min() < -1.0 or x.max() > 1.5:
        loss = F.binary_cross_entropy_with_logits(x, targets, reduction="none")
    else:
        # 概率输入可能因 sigmoid 饱和精确到 0/1, clamp 避免 -log(0)=inf → NaN
        x = x.clamp(1e-6, 1 - 1e-6)
        loss = F.binary_cross_entropy(x, targets, reduction="none")
    loss_flat, tgt_flat = loss.flatten(1), targets.flatten(1)
    ohem = []
    for b in range(x.shape[0]):
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


def ohem_masked_bce(pred, gt, mask, neg_ratio=3.0, eps=1e-6):
    """BalanceCrossEntropy（对照 PaddleOCR_DBNet）：OHEM + mask，pred 已过 sigmoid。

    positive = gt*mask, negative = (1-gt)*mask; 取全部正样本 + top-k 难负样本。
    """
    positive = gt * mask
    negative = (1 - gt) * mask
    positive_count = int(positive.sum())
    negative_count = min(int(negative.sum()), int(positive_count * neg_ratio))
    loss = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), gt, reduction="none")
    positive_loss = (loss * positive).sum()
    if negative_count > 0:
        # topk 后要 .sum(), 否则返回的是张量不是标量
        negative_loss = (loss * negative).reshape(-1).topk(negative_count).values.sum()
    else:
        negative_loss = loss.new_tensor(0.0)
    return (positive_loss + negative_loss) / (positive_count + negative_count + eps)


def dice_masked(pred, gt, mask, eps=1e-6):
    """Masked Dice（对照 PaddleOCR_DBNet）：binary 图的监督，带 mask。"""
    if pred.dim() == 4:
        pred = pred[:, 0]
    if gt.dim() == 4:
        gt = gt[:, 0]
    if mask.dim() == 4:
        mask = mask[:, 0]
    intersection = (pred * gt * mask).sum()
    union = (pred * mask).sum() + (gt * mask).sum() + eps
    return 1 - 2.0 * intersection / union


def db_loss(
    preds: dict,          # DBNetHead 输出: {prob(logits), thr(logits), binary(sigmoid)}
    shrink_map: torch.Tensor,    # (B,1,H,W) 收缩文本掩膜
    shrink_mask: torch.Tensor,   # (B,1,H,W) 有效区域掩膜
    threshold_map: torch.Tensor, # (B,1,H,W) 软阈值图 0.3~0.7
    threshold_mask: torch.Tensor,  # (B,1,H,W) 阈值监督区域掩膜
    alpha: float = 1.0,    # shrink 损失权重 (对照原版 α=1)
    beta: float = 10.0,    # threshold 损失权重 (对照原版 β=10)
    neg_ratio: float = 3.0,  # OHEM 负样本:正样本比例 (3=原版; 1=死区更易逃逸, 见下)
    eps: float = 1e-6,
):
    """DBNet 损失（对照 PaddleOCR_DBNet 原版）:
        loss = α·BalanceCE(prob, shrink) + β·MaskL1(thr, threshold) + Dice(binary, shrink)

    关键修复(对比之前版本):
        1. binary 用 Dice 监督、目标是收缩图 —— 之前用 BCE 对完整图, 和概率图目标冲突;
        2. 全程带 mask, 背景像素不直接主导;
        3. 阈值图是软值 0.3~0.7 而非 0/1 边界。

    neg_ratio 说明: OHEM 常数不动点在 p = 1/(1+neg_ratio) 处
        (neg_ratio=3 → p=0.25, binary=sigmoid(k·(P−T)) 在此完全饱和、梯度≈0, 训练死锁;
         neg_ratio=1 → p=0.5, 恰是 binary 分支梯度最大处, 死区更容易逃逸)。
    """
    prob, thr, binary = preds["prob"], preds["thr"], preds["binary"]
    prob_p = torch.sigmoid(prob)  # 概率图 → 概率
    thr_p = torch.sigmoid(thr)    # 阈值图 → 概率

    target = prob.shape[-2:]
    def rb(x):
        if x.shape[-2:] != target:
            x = F.interpolate(x.float(), size=target, mode="bilinear", align_corners=False)
        return x
    sm = rb(shrink_map); smask = rb(shrink_mask)
    tm = rb(threshold_map); tmask = rb(threshold_mask)

    loss_shrink = ohem_masked_bce(prob_p, sm, smask, neg_ratio=neg_ratio)  # 概率图: OHEM BCE + mask
    loss_thr = (torch.abs(thr_p - tm) * tmask).sum() / (tmask.sum() + eps)  # 阈值图: MaskL1
    loss_binary = dice_masked(binary, sm, smask)              # 二值图: Dice + mask

    total = alpha * loss_shrink + beta * loss_thr + loss_binary
    return {
        "total": total,
        "shrink": loss_shrink,
        "thr": loss_thr,
        "binary": loss_binary,
    }
