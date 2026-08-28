"""识别头训练 demo（EOS 版）：用 PIL 渲染随机数字串当"文本行图"，训练 RecognitionHead。

对比上一版(PAD 填充)的改进: 加了 EOS(结束)token——
短串多读的问题应该消失('9' → '9' 而不是 '9911')。
训练集/测试集分开，验证是学懂了还是背题。
"""

import random

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from heads import RecognitionHead
from heads.recognition_utils import build_vocab, encode_target, decode_prediction

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALPHABET = "0123456789"          # 10 个字符
MAX_LEN = 4                      # 最大字符数 = 查询 token 数
VOCAB = len(ALPHABET) + 1        # 10 字符 + EOS
IMG_H, IMG_W = 32, 128

CHAR_TO_IDX, IDX_TO_CHAR = build_vocab(ALPHABET)

# ---- 字体：优先 truetype(渲染更清晰)，失败退回默认位图字体 ----
FONT = None
for path in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/consola.ttf"]:
    try:
        FONT = ImageFont.truetype(path, 26)
        break
    except Exception:
        pass


def make_text_image(text, seed=None):
    """渲染一条文本行：白字深底，随机横向偏移。返回 (3,32,128) 张量。"""
    rng = random.Random(seed)
    img = Image.new("L", (IMG_W, IMG_H), 0)
    d = ImageDraw.Draw(img)
    x = rng.randint(2, 15)
    d.text((x, 3), text, fill=255, font=FONT)
    arr = np.array(img.convert("RGB"))
    return torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0)


def make_dataset(n, seed_offset=0):
    """生成 n 条随机数字串(长度 1~MAX_LEN) → (图像, 定长EOS标签)。"""
    images, targets = [], []
    for i in range(n):
        text = "".join(random.Random(seed_offset * 1000 + i).choices(ALPHABET, k=random.randint(1, MAX_LEN)))
        images.append(make_text_image(text, seed=i))
        targets.append(encode_target(text, CHAR_TO_IDX, MAX_LEN))
    return torch.stack(images), torch.tensor(targets)


def eval_metrics(batch_pred, batch_tgt):
    """按 EOS 停止解码后，算字符准确率和整串准确率。"""
    char_hit = char_total = str_hit = 0
    for p, t in zip(batch_pred, batch_tgt):
        p_str = decode_prediction(p, IDX_TO_CHAR)          # 遇到 EOS 停
        t_str = decode_prediction(t, IDX_TO_CHAR)
        if p_str == t_str:
            str_hit += 1
        # 字符级: 对齐比较(不等长只算到较短者)
        for a, b in zip(p_str, t_str):
            char_total += 1
            if a == b:
                char_hit += 1
    return char_hit / max(char_total, 1), str_hit / len(batch_pred)


def main():
    torch.manual_seed(0)
    print(f"字体: {FONT.path if FONT else '默认位图'}  设备: {DEVICE}")

    train_imgs, train_tgt = make_dataset(800, seed_offset=0)
    test_imgs, test_tgt = make_dataset(200, seed_offset=1000)
    print(f"训练集 {len(train_imgs)} 条, 测试集 {len(test_imgs)} 条 (不同字符串, EOS 词表)")

    BATCH = 32
    STEPS = 1200
    model = RecognitionHead(d_model=128, vocab_size=VOCAB, max_len=MAX_LEN).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    print(f"识别头参数量: {sum(p.numel() for p in model.parameters()):,}\n")

    for step in range(STEPS):
        model.train()
        idx = torch.randint(0, len(train_imgs), (BATCH,))
        x = train_imgs[idx].to(DEVICE)
        y = train_tgt[idx].to(DEVICE)
        opt.zero_grad()
        logits = model(x)  # (B, MAX_LEN, VOCAB)
        # 所有位置都有合法标签(字符或 EOS), 不需要 ignore_index
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        opt.step()
        sched.step()

        if step % 120 == 0 or step == STEPS - 1:
            model.eval()
            with torch.no_grad():
                tr_pred = model(train_imgs.to(DEVICE)).argmax(-1).cpu()
                te_pred = model(test_imgs.to(DEVICE)).argmax(-1).cpu()
            tr_char, tr_str = eval_metrics(tr_pred, train_tgt)
            te_char, te_str = eval_metrics(te_pred, test_tgt)
            print(f"step {step:3d}: loss={loss.item():.4f}  "
                  f"train 字符{tr_char:.2f}/整串{tr_str:.2f}  "
                  f"test 字符{te_char:.2f}/整串{te_str:.2f}")

    print("\n=== 测试集抽样 (EOS 停止解码) ===")
    with torch.no_grad():
        pred = model(test_imgs.to(DEVICE)).argmax(-1).cpu()
    shown = 0
    for i in range(len(test_imgs)):
        t_str = decode_prediction(test_tgt[i], IDX_TO_CHAR)
        p_str = decode_prediction(pred[i], IDX_TO_CHAR)
        print(f"  目标 '{t_str}'  预测 '{p_str}'  {'[OK]' if t_str == p_str else '[X]'}")
        shown += 1
        if shown >= 10:
            break


if __name__ == "__main__":
    main()
