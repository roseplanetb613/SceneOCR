"""识别头预训练：在线合成数据训练 RecognitionHead（EOS 并行解码）。

支持:
    - 断点保存/续训: 每 --save_every 步存一次完整 checkpoint(模型+优化器+调度器+step),
      训练被打断(Ctrl+C / 崩溃)后 --resume 续训;
    - 日志跟踪: 每次评估结果追加到 --log 文件, 崩溃也能看历史曲线;
    - 在线生成数据: 每步都是新图, 无限量, 无需恢复数据集状态。

用法:
    PYTHONPATH="E:/Sam2/ocr" python examples/pretrain_recognizer.py \
        --steps 3000 --batch 32 --ckpt checkpoints/recog_pretrain.pt \
        --log logs/pretrain.log --save_every 500
    # 续训(断点后):
    ... python examples/pretrain_recognizer.py --resume ...
"""

import argparse
import os
import random
import time

import torch
from torch.utils.data import DataLoader

from heads import RecognitionHead
from heads.recognition_utils import build_vocab, encode_target, decode_prediction
from data.synth import SynthTextLineDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ==================== checkpoint ====================
def save_checkpoint(path, model, opt, sched, step, best_str, args):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "step": step,
            "best_str": best_str,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path, model, opt, sched, device):
    """加载 checkpoint, 返回 (已训练的步数, best_str)。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    sched.load_state_dict(ckpt["scheduler"])
    return ckpt["step"], ckpt["best_str"]


# ==================== 日志 ====================
def log_line(path, text, echo=True):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    if echo:
        print(text)


# ==================== 评估 ====================
def eval_metrics(model, imgs, texts, idx_to_char, device, chunk=32):
    """验证集字符/整串准确率。按 chunk 分批, 避免一次塞太多进显存。"""
    model.eval()
    char_hit = char_total = str_hit = total = 0
    with torch.no_grad():
        for i in range(0, len(imgs), chunk):
            x = imgs[i : i + chunk].to(device)
            pred = model(x).argmax(-1).cpu()
            for p, t in zip(pred, texts[i : i + chunk]):
                p_str = decode_prediction(p, idx_to_char)
                if p_str == t:
                    str_hit += 1
                total += 1
                for a, b in zip(p_str, t):
                    char_total += 1
                    if a == b:
                        char_hit += 1
    return char_hit / max(char_total, 1), str_hit / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphabet", default=DEFAULT_ALPHABET)
    ap.add_argument("--max_len", type=int, default=8)
    ap.add_argument("--img_h", type=int, default=32)
    ap.add_argument("--img_w", type=int, default=128)
    ap.add_argument("--aug_level", type=int, default=2,
                    help="合成增强强度: 0=最小(单字体白字黑底) 1=轻 2=全量(默认)")
    ap.add_argument("--fixed_size", type=int, default=0,
                    help=">0 时预生成这么多张固定图反复训练(有轮次, 能学会); 0=在线生成(需百万级步数)")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_size", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--ckpt", default="checkpoints/recog_pretrain.pt")
    ap.add_argument("--log", default="logs/pretrain.log")
    ap.add_argument("--resume", action="store_true", help="从 --ckpt 续训")
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    VOCAB = len(args.alphabet) + 1  # + EOS
    char_to_idx, idx_to_char = build_vocab(args.alphabet)
    log_line(args.log, f"== 启动 (词表{len(args.alphabet)}+EOS={VOCAB}, 设备{DEVICE}, steps={args.steps}) ==")

    # ---- 数据（在线生成）----
    train_ds = SynthTextLineDataset(args.alphabet, args.max_len,
                                    num_samples=args.fixed_size or 1_000_000,
                                    img_h=args.img_h, img_w=args.img_w, seed=1,
                                    aug_level=args.aug_level, fixed=args.fixed_size > 0)
    val_ds = SynthTextLineDataset(args.alphabet, args.max_len, num_samples=args.val_size,
                                  img_h=args.img_h, img_w=args.img_w, seed=999,
                                  aug_level=args.aug_level, fixed=True)
    val_imgs = torch.stack([img for img, _ in val_ds])  # 预生成, 跨轮次可比
    val_texts = [t for _, t in val_ds]

    # ---- 模型 / 优化器 / 调度器 ----
    model = RecognitionHead(d_model=args.d_model, vocab_size=VOCAB,
                            max_len=args.max_len).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    start_step, best_str = 0, 0.0
    if args.resume and os.path.exists(args.ckpt):
        start_step, best_str = load_checkpoint(args.ckpt, model, opt, sched, DEVICE)
        log_line(args.log, f"== 从 {args.ckpt} 续训 (step {start_step}, best_str {best_str:.3f}) ==")
    print(f"识别头参数量: {sum(p.numel() for p in model.parameters()):,}")

    mode = f"固定{args.fixed_size}张(有轮次)" if args.fixed_size > 0 else "在线无限(需海量步数)"
    log_line(args.log, f"数据模式: {mode}  aug_level={args.aug_level}")
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
            y = torch.tensor([encode_target(t, char_to_idx, args.max_len) for t in texts])
            opt.zero_grad()
            logits = model(imgs.to(DEVICE))
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, VOCAB), y.to(DEVICE).reshape(-1)
            )
            loss.backward()
            opt.step()
            sched.step()

            # ---- 定期评估 + 日志 + 存 best ----
            if step % args.log_every == 0 or step == args.steps - 1:
                if DEVICE.startswith("cuda"):
                    torch.cuda.empty_cache()
                char, string = eval_metrics(model, val_imgs, val_texts, idx_to_char, DEVICE)
                spd = (step + 1 - start_step) / max(time.time() - t0, 1e-6)
                log_line(args.log, f"step {step:5d}: loss={loss.item():.4f} "
                         f"val 字符{char:.3f}/整串{string:.3f}  ({spd:.1f} 步/s)")
                if string > best_str:
                    best_str = string
                    save_checkpoint(args.ckpt, model, opt, sched, step + 1, best_str, args)
                    log_line(args.log, f"  → 新 best 整串{string:.3f}, 已存 {args.ckpt}")

            # ---- 定期存断点（不管有没有变 best, 保证可续训）----
            if (step + 1) % args.save_every == 0:
                save_checkpoint(args.ckpt, model, opt, sched, step + 1, best_str, args)
                log_line(args.log, f"  → 断点已存 step {step+1}")

    except KeyboardInterrupt:
        # Ctrl+C: 存断点再退出, 不丢进度
        save_checkpoint(args.ckpt, model, opt, sched, step + 1, best_str, args)
        log_line(args.log, f"== 被中断, 断点已存 step {step+1} ==")
        return

    log_line(args.log, f"== 完成. best 整串准确率: {best_str:.3f}, checkpoint: {args.ckpt} ==")


if __name__ == "__main__":
    main()
