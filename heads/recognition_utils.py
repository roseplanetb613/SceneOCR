"""识别头的词表 / 编码 / 解码工具（EOS 版）。

解决并行解码器的"多读"问题:
    并行解码有 max_len 个固定查询, 短文本时多余的查询会乱猜。
    加一个 EOS(结束)token 后, 目标串末尾用 EOS 补齐,
    模型学会"到这输出 EOS = 后面没了", 解码时遇到 EOS 就停。
    这样短串 '9' 会解码成 '9', 而不是 '9911'。

词表构成: [字母表...] + EOS
    EOS 既当"结束符"也当"填充符"(设计 A): 所有位置都有合法标签, 损失不需要 ignore_index。
"""

EOS = "<eos>"


def build_vocab(alphabet: str):
    """根据字母表构造 字符↔索引 映射。EOS 放在最后。"""
    char_to_idx = {c: i for i, c in enumerate(alphabet)}
    char_to_idx[EOS] = len(alphabet)
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    return char_to_idx, idx_to_char


def encode_target(text: str, char_to_idx: dict, max_len: int):
    """字符串 → 定长标签序列 [c1..cn, EOS, EOS, ...]，长度 = max_len。"""
    idxs = [char_to_idx[c] for c in text if c in char_to_idx][:max_len]
    eos = char_to_idx[EOS]
    return idxs + [eos] * (max_len - len(idxs))


def decode_prediction(argmax_idxs, idx_to_char: dict):
    """argmax 索引序列 → 字符串（遇到 EOS 停止）。"""
    out = []
    for idx in argmax_idxs:
        c = idx_to_char[int(idx)]
        if c == EOS:
            break
        out.append(c)
    return "".join(out)
