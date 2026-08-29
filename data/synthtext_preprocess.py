"""SynthText 预处理：把 1.9GB 的 gt.mat 转成轻量缓存（只用 wordBB + 图名）。

gt.mat 里有 charBB / wordBB / imnames / txt 四块, 检测只需要 wordBB(词级四点框) + imnames。
这里只加载这两块, 提取成 [图名, 每图多边形列表] 的 pickle 缓存,
避免训练时每次重复读 1.9GB 的 mat(还占内存)。

用法:
    python data/synthtext_preprocess.py --root <SynthText路径> --cache <缓存输出路径>
"""

import argparse
import os
import pickle
import time

import numpy as np
import scipy.io as sio


def parse_wordbb(wb):
    """把单张图的 wordBB 变成 (N,4,2) float32 多边形列表。

    wordBB 常见形状 (2,4,N): [x;y] × 4 角 × N 词。
    个别是 (2,4,N,M): 取第一个 M(整体框)。
    """
    if wb.ndim == 3:
        return wb.transpose(2, 1, 0).astype(np.float32)   # (N,4,2)
    elif wb.ndim == 4:
        return wb[:, :, :, 0].transpose(2, 1, 0).astype(np.float32)
    elif wb.shape == (2, 4):
        return wb.T[None].astype(np.float32)              # 单词 → (1,4,2)
    raise ValueError(f"Unexpected wordBB shape {wb.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="E:/迅雷下载/SynthText/SynthText")
    ap.add_argument("--cache", default="data/cache_synthtext.pkl")
    args = ap.parse_args()

    t0 = time.time()
    # 只加载 wordBB + imnames, 跳过 charBB(大) 和 txt
    m = sio.loadmat(os.path.join(args.root, "gt.mat"),
                    variable_names=["wordBB", "imnames"])
    wordBB = m["wordBB"][0]      # (858750,) object array, 每项 (2,4,N)
    imnames = m["imnames"][0]    # (858750,) object array of str
    print(f"gt.mat 加载完成: {len(imnames)} 张图, 耗时 {time.time()-t0:.0f}s")

    names, polys_list = [], []
    empty = missing = 0
    for i in range(len(imnames)):
        name = str(imnames[i][0])
        # SynthText 有少量"标注存在但图缺失"的坏数据, 过滤掉
        if not os.path.exists(os.path.join(args.root, name)):
            missing += 1
            continue
        polys = parse_wordbb(wordBB[i])
        if len(polys) == 0:
            empty += 1
        names.append(name)
        polys_list.append(polys)
    print(f"提取完成: {len(names)} 张, 无词框的 {empty} 张, 图缺失的 {missing} 张(已过滤)")

    # 存缓存
    os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
    with open(args.cache, "wb") as f:
        pickle.dump({"names": names, "polys": polys_list}, f)
    size_mb = os.path.getsize(args.cache) / 1e6
    print(f"缓存已存: {args.cache} ({size_mb:.0f} MB), 总耗时 {time.time()-t0:.0f}s")
    print(f"样例: {names[0]}, {len(polys_list[0])} 个词")


if __name__ == "__main__":
    main()
