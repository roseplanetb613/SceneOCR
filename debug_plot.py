"""生成「死区自锁 vs 修复后」面试展示对比图 (docs/deadlock_vs_fix.png)。

数据来源: logs/det_td500_v2.log (neg_ratio=3, 死锁) 与
         logs/det_td500_neg1.log + neg1b.log (neg_ratio=1, 续跑, step 偏移 +1000)。
"""
import re
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ---- 中文字体 ----
for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "SimSun"]:
    if any(f.name == name for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [name]
        break
plt.rcParams["axes.unicode_minus"] = False


def parse_log(path):
    """从训练日志提取 (step, shrink, thr, binary, val_iou)。"""
    steps, shrinks, thrs, binaries, ious = [], [], [], [], []
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = re.search(
            r"step\s+(\d+):.*?shrink=([\d.]+), thr=([\d.]+), binary=([\d.]+).*?val-binary-IoU=([\d.]+)",
            line)
        if m:
            steps.append(int(m.group(1)))
            shrinks.append(float(m.group(2)))
            thrs.append(float(m.group(3)))
            binaries.append(float(m.group(4)))
            ious.append(float(m.group(5)))
    return steps, shrinks, thrs, binaries, ious


# ---- 真实数据 ----
s3, _, _, b3, i3 = parse_log("logs/det_td500_v2.log")       # neg_ratio=3: 死锁
s1, _, _, b1, i1 = parse_log("logs/det_td500_neg1.log")      # neg_ratio=1: 1e-4, 0~1000
s1b, _, _, b1b, i1b = parse_log("logs/det_td500_neg1b.log")  # neg_ratio=1: 1e-3, 续跑偏移+1000
s1b = [x + 1000 for x in s1b]

fig = plt.figure(figsize=(15, 11), dpi=160)
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.2,
                      left=0.09, right=0.95, top=0.9, bottom=0.1)

# ============ (0,0) 验证 IoU 对比 ============
ax = fig.add_subplot(gs[0, 0])
ax.plot(s3, i3, "o-", color="#c0392b", lw=2, ms=6, label="neg_ratio=3（修复前 · 死锁）")
ax.plot(s1, i1, "s-", color="#2980b9", lw=2, ms=6, label="neg_ratio=1（修复后）")
ax.plot(s1b, i1b, "s-", color="#2980b9", lw=2, ms=4, alpha=0.75,
        label="neg_ratio=1（续跑 lr=1e-3）")
ax.axhline(0, color="#c0392b", ls="--", lw=1.2, alpha=0.6)
ax.annotate("永远 0.000", xy=(1050, 0.002), color="#c0392b", fontsize=13, fontweight="bold")
ax.annotate("开始学习", xy=(230, 0.055), color="#2980b9", fontsize=13, fontweight="bold")
ax.set_xlabel("训练步数", fontsize=12)
ax.set_ylabel("验证二值掩膜 IoU", fontsize=12)
ax.set_title("训练曲线对比：死锁 vs 修复（TD500，实测）", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="upper left")
ax.grid(alpha=0.3)

# ============ (0,1) binary 损失对比 ============
ax = fig.add_subplot(gs[0, 1])
ax.plot(s3, b3, "o-", color="#c0392b", lw=2, ms=6, label="neg_ratio=3：恒 1.000（分支死亡）")
ax.plot(s1, b1, "s-", color="#2980b9", lw=2, ms=6, label="neg_ratio=1：0.60~0.96（分支复活）")
ax.plot(s1b, b1b, "s-", color="#2980b9", lw=2, ms=4, alpha=0.75)
ax.set_ylim(0.5, 1.05)
ax.axhline(1.0, color="#c0392b", ls="--", lw=1.2, alpha=0.6)
ax.annotate("Dice=1.0 → 二值图全零 → 梯度≈0", xy=(300, 0.995),
            color="#c0392b", fontsize=12, fontweight="bold")
ax.annotate("梯度恢复传导", xy=(1200, 0.72), color="#2980b9",
            fontsize=12, fontweight="bold")
ax.set_xlabel("训练步数", fontsize=12)
ax.set_ylabel("binary 损失（Dice）", fontsize=12)
ax.set_title("二值分支死活对比（binary 损失）", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.grid(alpha=0.3)

# ============ (1,0) 机制：sigmoid(k·z) 的梯度 vs P−T ============
ax = fig.add_subplot(gs[1, 0])
k = 10.0
z = np.linspace(-1.5, 1.5, 400)
binary = 1 / (1 + np.exp(-k * z))
grad = k * binary * (1 - binary)          # d(binary)/d(P−T)
ax.plot(z, binary, "-", color="#7f8c8d", lw=2, label="binary = sigmoid(k·(P−T))")
ax2 = ax.twinx()
ax2.plot(z, grad, "-", color="#e67e22", lw=2.5, label="|梯度| = k·binary·(1−binary)")
# 标注两个关键点
z_dead, z_alive = -1.1, 0.0               # p=0.25 / p=0.5 时 P−T ≈ z
ax.axvline(z_dead, color="#c0392b", ls="--", lw=1.2)
ax.axvline(z_alive, color="#27ae60", ls="--", lw=1.2)
ax.plot(z_dead, 1/(1+np.exp(-k*z_dead)), "o", color="#c0392b", ms=10, zorder=5)
ax.plot(z_alive, 0.5, "o", color="#27ae60", ms=10, zorder=5)
ax.annotate("P−T ≈ −1.1\n(p=0.25)\n梯度 ≈ 0  →  死锁", xy=(z_dead, 0.5),
            xytext=(z_dead-1.25, 0.78), fontsize=12, fontweight="bold",
            color="#c0392b", arrowprops=dict(arrowstyle="->", color="#c0392b"))
ax.annotate("P−T ≈ 0\n(p=0.5)\n梯度 = k·0.25 = 2.5  →  复活", xy=(z_alive, 0.5),
            xytext=(0.35, 0.95), fontsize=12, fontweight="bold",
            color="#27ae60", arrowprops=dict(arrowstyle="->", color="#27ae60"))
ax.set_xlabel("P − T（概率图 logits − 阈值图 logits）", fontsize=12)
ax.set_ylabel("binary 值", fontsize=12, color="#7f8c8d")
ax2.set_ylabel("梯度幅值", fontsize=12, color="#e67e22")
ax.set_title("机制：为什么 p=0.25 死锁、p=0.5 复活", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.05)
ax2.set_ylim(0, 3.0)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="center left")
ax.grid(alpha=0.3)

# ============ (1,1) 不动点 p=1/(1+r) 与梯度 ============
ax = fig.add_subplot(gs[1, 1])
r = np.linspace(0.5, 5.0, 200)
p = 1 / (1 + r)                            # OHEM 常数不动点
z_p = np.log(p / (1 - p))                  # 不动点处的 P logits(设 T≈0)
g_p = k * (1/(1+np.exp(-k*z_p))) * (1 - 1/(1+np.exp(-k*z_p)))
ax.plot(r, p, "-", color="#2980b9", lw=2.5, label="不动点 p = 1/(1+neg_ratio)")
ax2 = ax.twinx()
ax2.plot(r, g_p, "-", color="#e67e22", lw=2.5, label="binary 分支梯度幅值")
# 标注 r=3 与 r=1
ax.axvline(3.0, color="#c0392b", ls="--", lw=1.2)
ax.axvline(1.0, color="#27ae60", ls="--", lw=1.2)
ax.plot(3.0, 0.25, "o", color="#c0392b", ms=10, zorder=5)
ax.plot(1.0, 0.5, "o", color="#27ae60", ms=10, zorder=5)
ax.annotate("r=3 → p=0.25\n梯度≈0（死锁）", xy=(3.0, 0.25), xytext=(3.35, 0.16),
            fontsize=12, fontweight="bold", color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b"))
ax.annotate("r=1 → p=0.5\n梯度最大（复活）", xy=(1.0, 0.5), xytext=(0.62, 0.62),
            fontsize=12, fontweight="bold", color="#27ae60",
            arrowprops=dict(arrowstyle="->", color="#27ae60"))
ax.set_xlabel("neg_ratio（OHEM 负:正样本比）", fontsize=12)
ax.set_ylabel("常数不动点 p", fontsize=12, color="#2980b9")
ax2.set_ylabel("binary 梯度幅值", fontsize=12, color="#e67e22")
ax.set_title("一行参数 = 算出来的修复方向", fontsize=14, fontweight="bold")
ax.set_xlim(0.5, 5)
ax.set_ylim(0, 0.75)
ax2.set_ylim(0, 3.0)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower left")
ax.grid(alpha=0.3)

fig.suptitle("DBNet 检测头「死区死锁」根因与一行参数修复",
             fontsize=17, fontweight="bold", y=0.98)
os.makedirs("docs", exist_ok=True)
fig.savefig("docs/deadlock_vs_fix.png", bbox_inches="tight", facecolor="white")
print("已保存: docs/deadlock_vs_fix.png")
