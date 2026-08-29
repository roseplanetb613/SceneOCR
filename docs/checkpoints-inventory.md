# 权重清单（Checkpoint Inventory）

> 所有权重按「位置」分类：`checkpoints/` 根目录 = **当前可用**（脚本默认引用），
> `checkpoints/history_detector/` 与 `checkpoints/history_recognizer/` = **排查历史证据**。
> 历史权重**刻意保留不删**——它们是调试过程的"来路"，每个都对应一段实验结论。
> 训练日志在 `logs/`（按运行时间可回溯）。

## ✅ 当前可用（`checkpoints/` 根目录）

| 文件 | 用途 | 训练配置 | 指标 |
|---|---|---|---|
| `det_synth_v2.pt` | 检测：合成预训练 | 冻结骨干 + 预训练 Hiera，合成整图 2000 步 | val IoU **0.847** |
| `det_td500_neg1b.pt` | 检测：TD500 微调（全冻结） | `--init_ckpt det_synth_v2.pt --neg_ratio 1`，3000 步 | 确定性 mean IoU 0.060 |
| `det_td500_neck2.pt` | 检测：TD500 微调（合成+微调都训 neck） | `--train_neck` 两阶段 | 确定性 mean IoU 0.056（三组对照之一）|
| `ctc_synth62.pt` | 识别：合成预训练 | 62 词表，合成 10000 张，8000 步 | 整串 **95.5%** |
| `ctc_real62c.pt` | 识别：SynthText 真实微调（**推荐**） | `--init_ckpt ctc_synth62.pt`，150k 词，12000 步 | 整串 **60.6%** / 字符 76.2% |

**推荐链路**：检测 = `det_synth_v2.pt`（预训练）→ `det_td500_neg1b.pt`（微调）；
识别 = `ctc_synth62.pt`（合成）→ `ctc_real62c.pt`（真实）。推理脚本默认即指向可用权重。

## 📜 检测历史（`checkpoints/history_detector/`）—— 排查来路

| 文件 | 时期 | 对应实验结论 |
|---|---|---|
| `det_synth.pt` | 8/28 | 最早合成预训练（旧损失/旧头结构），**IoU 0.943**——证明"头能学"，把嫌疑引向数据/配置 |
| `det_td500.pt` | 8/28 | 旧版 TD500（当时数据管线有 bug） |
| `det_synthtext.pt` | 8/29 | 旧损失下 SynthText：`shrink` 卡 0.56，**死区早期症状** |
| `det_finetuned.pt` | 8/28 | 旧权重微调 TD500：IoU 0.000~0.08 |
| `det_fix.pt` | 8/29 | OHEM 修复尝试：`prob_bce` 卡 0.564 |
| `det_k10.pt` | 8/29 | k=10 调整尝试：同样卡死 |
| `det_st.pt` | 8/29 | SynthText 训练：IoU 0.000 |
| `det_v2.pt` | 8/29 | 死区"全文本"模式：`shrink` 卡 10.5 |
| `det_v3.pt` | 8/29 | 无预训练骨干 + train_backbone：卡 3.46 |
| `det_pretrained.pt` | 8/29 | 预训练骨干 + train_backbone：仍死区 |
| `det_gn.pt` | 8/29 | GroupNorm 头 + train_backbone：`shrink` 0.556 死锁 |
| `det_long.pt` | 8/29 | SynthText 长训 + train_backbone：**4400 步 IoU 恒 0.000（死区实证）** |
| `det_resume_test.pt` | 8/29 | 断点续训机制冒烟测试 |
| `det_synth_neck.pt` | 8/29 | 合成预训练 + `--train_neck`（正确姿势实验的合成阶段）|
| `det_td500_neck.pt` | 8/29 | 仅微调解冻 neck（从零学 FPN → 未超基线）|

## 📜 识别历史（`checkpoints/history_recognizer/`）—— 完善来路

| 文件 | 时期 | 对应实验结论 |
|---|---|---|
| `ctc_digits.pt` | 8/28 | 10 数字词表，**整串 98.5%**（识别最早里程碑）|
| `ctc_full.pt` | 8/28 | 36 字符词表，**整串 94.5%**（旧版 README 引用）|
| `recog_easy.pt` / `recog_easyA.pt` / `recog_fixed.pt` / `recog_pretrain.pt` | 8/28 | **RecognitionHead（并行解码版，已弃用）** 的训练产物，对比证明 CTC 更适合 |
| `ctc_real62.pt` | 8/29 | 真实微调 v1：轴对齐裁剪 + 270k 词 → **21.5%**（证明裁剪方式不行）|
| `ctc_real62b.pt` | 8/29 | 真实微调 v2：透视矫正 + 60k 词 → **40.9%**（裁剪修复的量化证据）|

## 决策树（面试/复盘用）

- **为什么历史权重全保留？** 每个文件对应一条实验结论——"来路"本身就是排查能力的证据：
  `det_synth.pt`（头能学）→ `det_long.pt`（死区实证）→ `det_td500_neg1b.pt`（`neg_ratio=1` 修复后）；
  `ctc_real62.pt`（轴对齐 21.5%）→ `ctc_real62b.pt`（透视 40.9%）→ `ctc_real62c.pt`（150k 数据 60.6%）。
- **为什么根目录只留 5 个？** 它们是被脚本默认引用 / 当前推荐链路的权重，保证开箱即用。
