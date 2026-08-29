"""CTC 识别头预训练：合成数据训练 CTCHead（CTC 对齐，自动处理变长/对齐）。

和并行解码器(RecognitionHead)的区别:
    并行解码器靠固定查询位置对齐, 位置一随机就崩;
    CTC 让每一列特征独立打分, ctc_loss 自动对齐, 对随机偏移/旋转健壮。

用法:
    PYTHONPATH="E:/Sam2/ocr" python examples/pretrain_ctc.py \
        --alphabet "0123456789" --max_len 8 --aug_level 2 \
        --fixed_size 3000 --steps 3000 --batch 32 \
        --ckpt checkpoints/ctc_pretrain.pt --log logs/ctc_pretrain.log
    # 断点续训:
    ... python examples/pretrain_ctc.py --resume ...
"""

import argparse
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from heads import CTCHead
from heads.ctc_utils import encode_text, ctc_greedy_decode
from heads.recognition_utils import build_vocab
from data.synth import SynthTextLineDataset
from data.synthtext_rec import (SynthTextRecDataset, PrecroppedRecDataset,
                                IMG_H, IMG_W)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 62 字符: 数字 + 大写 + 小写 (覆盖 SynthText 真实字符分布)
DEFAULT_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


# ==================== checkpoint / 日志 ====================
def save_checkpoint(path, model, opt, sched, step, best_str, args):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": opt.state_dict(),
         "scheduler": sched.state_dict(), "step": step,
         "best_str": best_str, "args": vars(args)}, path)


def load_checkpoint(path, model, opt, sched, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    sched.load_state_dict(ckpt["scheduler"])
    return ckpt["step"], ckpt["best_str"]


def log_line(path, text, echo=True):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    if echo:
        print(text)


# ==================== 评估 ====================
def eval_metrics(model, imgs, texts, idx_to_char, blank_idx, device, chunk=32):
    model.eval()
    char_hit = char_total = str_hit = total = 0
    with torch.no_grad():
        for i in range(0, len(imgs), chunk):
            logits = model(imgs[i : i + chunk].to(device))
            preds = ctc_greedy_decode(logits, idx_to_char, blank_idx)
            for p, t in zip(preds, texts[i : i + chunk]):
                if p == t:
                    str_hit += 1
                total += 1
                for a, b in zip(p, t):
                    char_total += 1
                    if a == b:
                        char_hit += 1
    return char_hit / max(char_total, 1), str_hit / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphabet", default=DEFAULT_ALPHABET)
    ap.add_argument("--data", choices=["synth", "synthtext_rec"], default="synth",
                    help="synth=自研合成; synthtext_rec=SynthText 词级真实渲染裁剪")
    ap.add_argument("--synthtext_root", default="E:/迅雷下载/SynthText/SynthText")
    ap.add_argument("--synthtext_cache", default="data/cache_synthtext_rec.pkl")
    ap.add_argument("--synthtext_max", type=int, default=270000,
                    help="synthtext_rec 训练子集大小(270000≈1轮/8437步; 微调建议 60000 加速轮次)")
    ap.add_argument("--precrop_train", default="data/crops_rec_train.npy")
    ap.add_argument("--precrop_train_labels", default="data/labels_rec_train.pkl")
    ap.add_argument("--precrop_val", default="data/crops_rec_val.npy")
    ap.add_argument("--precrop_val_labels", default="data/labels_rec_val.pkl")
    ap.add_argument("--max_len", type=int, default=8)
    ap.add_argument("--img_h", type=int, default=32)
    ap.add_argument("--img_w", type=int, default=128)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--aug_level", type=int, default=2)
    ap.add_argument("--fixed_size", type=int, default=3000,
                    help=">0 固定数据集(有轮次); 0 在线无限")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_size", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=300)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--ckpt", default="checkpoints/ctc_pretrain.pt")
    ap.add_argument("--log", default="logs/ctc_pretrain.log")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init_ckpt", default=None,
                    help="只载入权重(优化器重置), 用于合成预训练 → 真实数据微调")
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    VOCAB = len(args.alphabet)
    char_to_idx, idx_to_char = build_vocab(args.alphabet)
    blank_idx = VOCAB  # blank 是词表后第一个索引

    mode = f"固定{args.fixed_size}张" if args.fixed_size > 0 else "在线无限"
    log_line(args.log, f"== CTC 启动 (data={args.data} 词表{VOCAB}+blank, "
             f"设备{DEVICE}, {mode}, aug={args.aug_level}) ==")

    # ---- 数据 ----
    if args.data == "synth":
        train_ds = SynthTextLineDataset(args.alphabet, args.max_len,
                                        num_samples=args.fixed_size or 1_000_000,
                                        img_h=args.img_h, img_w=args.img_w, seed=1,
                                        aug_level=args.aug_level, fixed=args.fixed_size > 0)
        val_ds = SynthTextLineDataset(args.alphabet, args.max_len, num_samples=args.val_size,
                                      img_h=args.img_h, img_w=args.img_w, seed=999,
                                      aug_level=args.aug_level, fixed=True)
    else:  # synthtext_rec (优先预裁剪缓存, 训练零读图开销)
        import os as _os
        if _os.path.exists(args.precrop_train):
            train_ds = PrecroppedRecDataset(args.precrop_train, args.precrop_train_labels,
                                            aug=True, seed=1)
            val_ds = PrecroppedRecDataset(args.precrop_val, args.precrop_val_labels)
        else:
            train_ds = SynthTextRecDataset(args.synthtext_root, cache=args.synthtext_cache,
                                           max_items=args.synthtext_max, seed=1, aug=True)
            val_ds = SynthTextRecDataset(args.synthtext_root, cache=args.synthtext_cache,
                                         max_items=args.val_size, seed=999)
    val_imgs = torch.stack([img for img, _ in val_ds])
    val_texts = [t for _, t in val_ds]

    # ---- 模型 ----
    model = CTCHead(d_model=args.d_model, vocab_size=VOCAB, img_h=args.img_h).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    start_step, best_str = 0, 0.0
    if args.resume and os.path.exists(args.ckpt):
        start_step, best_str = load_checkpoint(args.ckpt, model, opt, sched, DEVICE)
        log_line(args.log, f"== 从 {args.ckpt} 续训 (step {start_step}) ==")
    elif args.init_ckpt and os.path.exists(args.init_ckpt):
        ckpt = torch.load(args.init_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        log_line(args.log, f"== 从 {args.init_ckpt} 载入权重(优化器重置, 微调) ==")
    print(f"CTC 头参数量: {sum(p.numel() for p in model.parameters()):,}")

    loader = DataLoader(train_ds, batch_size=args.batch, num_workers=0, shuffle=True)
    it = iter(loader)
    t0 = time.time()

    try:
        for step in range(start_step, args.steps):
            try:
                imgs, texts = next(it)
            except StopIteration:
                it = iter(loader)
                imgs, texts = next(it)
            model.train()
            # CTC targets: 展平的目标索引 + 每样本长度
            targets = [encode_text(t, char_to_idx) for t in texts]
            flat = torch.tensor([i for tl in targets for i in tl], dtype=torch.long)
            target_lengths = torch.tensor([len(tl) for tl in targets], dtype=torch.long)
            opt.zero_grad()
            logits = model(imgs.to(DEVICE))            # (B, T, V+1)
            T = logits.size(1)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V+1)
            input_lengths = torch.full((logits.size(0),), T, dtype=torch.long)
            loss = F.ctc_loss(
                log_probs, flat.to(DEVICE),
                input_lengths.to(DEVICE), target_lengths.to(DEVICE),
                blank=blank_idx, zero_infinity=True,
            )
            loss.backward()
            opt.step()
            sched.step()

            if step % args.log_every == 0 or step == args.steps - 1:
                if DEVICE.startswith("cuda"):
                    torch.cuda.empty_cache()
                char, string = eval_metrics(model, val_imgs, val_texts, idx_to_char,
                                            blank_idx, DEVICE)
                spd = (step + 1 - start_step) / max(time.time() - t0, 1e-6)
                log_line(args.log, f"step {step:5d}: loss={loss.item():.4f} "
                         f"val 字符{char:.3f}/整串{string:.3f}  ({spd:.1f} 步/s)")
                if string > best_str:
                    best_str = string
                    save_checkpoint(args.ckpt, model, opt, sched, step + 1, best_str, args)
                    log_line(args.log, f"  → 新 best 整串{string:.3f}, 已存 {args.ckpt}")

            if (step + 1) % args.save_every == 0:
                save_checkpoint(args.ckpt, model, opt, sched, step + 1, best_str, args)
                log_line(args.log, f"  → 断点已存 step {step+1}")

    except KeyboardInterrupt:
        save_checkpoint(args.ckpt, model, opt, sched, step + 1, best_str, args)
        log_line(args.log, f"== 被中断, 断点已存 step {step+1} ==")
        return

    # 正常跑完: 存最终断点
    save_checkpoint(args.ckpt, model, opt, sched, args.steps, best_str, args)
    log_line(args.log, f"== 完成. best 整串准确率: {best_str:.3f}, checkpoint: {args.ckpt} ==")


if __name__ == "__main__":
    main()
