"""CTC 的编码 / 解码工具。

编码: 文本 → 字符索引序列(无 EOS、无 padding; CTC 靠 blank + 长度对齐)。
解码: 贪心解码 —— 每列取 argmax → 去掉 blank → 合并相邻重复字符。
"""


def encode_text(text: str, char_to_idx: dict):
    """文本 → 字符索引列表（长度 = 真实字符数, 不做 padding）。"""
    return [char_to_idx[c] for c in text if c in char_to_idx]


def ctc_greedy_decode(logits, idx_to_char: dict, blank_idx: int):
    """贪心解码: logits (B, T, V) → 每行一个字符串。

    规则: 每时间步取 argmax → 跳过 blank → 合并相邻重复字符。
    例: 序列 [2, blank, 2, 5] → "22"（blank 隔开的两个 2 都保留）
    """
    argmax = logits.argmax(-1)  # (B, T)
    texts = []
    for seq in argmax:
        out = []
        prev = -1
        for idx in seq.tolist():
            if idx != blank_idx and idx != prev:
                out.append(idx_to_char[idx])
            prev = idx
        texts.append("".join(out))
    return texts
