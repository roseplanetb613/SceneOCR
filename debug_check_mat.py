"""验证 charBB 字符级对齐: strip 后 txt 字符总数 == charBB M 的比例。

若对齐率高, 就用官方标准做法: 从 charBB 按 txt 分词重建词级 (裁剪图, 标签)。
"""
import scipy.io as sio
import numpy as np

root = "E:/迅雷下载/SynthText/SynthText"
m = sio.loadmat(root + "/gt.mat", variable_names=["wordBB", "charBB", "txt"])
charBB = m["charBB"][0]
txt = m["txt"][0]

N_SAMPLE = 40000
aligned = 0
mismatch = 0
total_imgs = 0
mismatch_examples = []

for i in range(N_SAMPLE):
    t = txt[i]
    cb = charBB[i]
    if t is None or len(t) == 0 or cb is None:
        continue
    total_imgs += 1
    n_chars = cb.shape[2] if cb.ndim == 3 else 1
    # strip 每个元素后拼接字符总数
    n_txt_chars = sum(len(str(t[j]).strip()) for j in range(len(t)))
    if n_txt_chars == n_chars:
        aligned += 1
    else:
        mismatch += 1
        if len(mismatch_examples) < 8:
            mismatch_examples.append(
                (i, n_chars, n_txt_chars,
                 [str(t[j])[:20] for j in range(min(3, len(t)))]))

print(f"抽样 {total_imgs} 图 (含 txt+charBB):")
print(f"  charBB M == strip后字符数: {aligned} ({aligned/max(total_imgs,1)*100:.1f}%)")
print(f"  不对齐: {mismatch} ({mismatch/max(total_imgs,1)*100:.1f}%)")
print("\n不对齐示例 (图, charBB数, txt字符数, txt前3项):")
for ex in mismatch_examples:
    print("  ", ex)
