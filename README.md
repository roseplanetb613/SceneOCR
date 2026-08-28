# OCR 引擎：场景文本检测 + 识别

一个从零实现的端到端 OCR 系统，覆盖 **文本检测（DBNet）** 与 **文本识别（CTC）** 两条主线，
并配套完整的合成数据管线、训练基础设施（断点续训/日志/评估）与可视化推理。

目标是**理解每一层、自己动手搭**，而不是套现成模型。视觉特征部分参考了近年主流的
层次化视觉 Transformer 设计，检测/识别头、数据生成、训练管线均为本项目自研。

---

## 亮点

- **双头 OCR 架构**：DBNet 检测头（概率图 + 阈值图 + 可微分二值化）+ CTC 识别头（特征序列对齐）
- **检测/识别都从零训练**：合成数据预训练 + 真实数据微调，标准配方
- **合成数据管线**：多字体/背景/旋转/噪声/模糊的文本行生成、整图文本框生成，在线或固定数据两用
- **稳健的训练基础设施**：断点保存/续训、日志跟踪、验证集评估、随时可中断不丢进度
- **组件化设计**：骨干、特征金字塔、注意力、通用块、双头全部解耦，可单独替换升级

## 系统架构

```
                     ┌──────────────────────────────────────┐
     整图 ──────────►│        文本检测（DBNet）              │
                     │  骨干 → FPN → DBNetHead              │
                     │  概率图 + 阈值图 → 可微分二值化        │
                     └──────────────┬───────────────────────┘
                                    │ 文本框 / 文本区域掩膜
                                    ▼
                     ┌──────────────────────────────────────┐
     文本行裁剪 ────►│        文本识别（CTC）                │
                     │  卷积骨干 → 序列编码 → 逐列打分       │
                     │  CTC 对齐 → 变长文字输出              │
                     └──────────────┬───────────────────────┘
                                    ▼
                               "OCR2026"
```

- **视觉特征提取**：自实现的层次化多尺度骨干 + 特征金字塔（FPN），输出多分辨率特征供检测头使用。
- **DBNet 检测头**：可微分二值化（DB）把"阈值"也变成网络输出，整条检测流程端到端可训；
  训练用 OHEM 难例挖掘解决文本像素占比过小的类别不平衡。
- **CTC 识别头**：不依赖固定位置对齐，对特征序列每列独立打分，`ctc_loss` 自动对齐变长文本，
  对随机偏移/旋转天然健壮。

## 模块一览

```
ocr/
├── components/          # 通用组件库（自实现）
│   ├── backbone/        # 层次化多尺度视觉骨干（Hiera 风格）
│   ├── neck/            # 特征金字塔 FPN + 位置编码
│   ├── attention/       # 多头注意力 / RoPE / 双向 Transformer / MLP
│   └── blocks/          # ConvNeXt 块 / LayerNorm2d / DropPath / 工具函数
├── heads/               # OCR 双头
│   ├── detection.py     # DBNetHead：概率图 + 阈值图 + 可微分二值化
│   ├── losses.py        # DBNet 损失（L_s 概率图 + L_t 阈值图 + L_b 二值图 + OHEM）
│   ├── recognition_ctc.py  # CTCHead：卷积骨干 + 序列编码 + 逐列分类
│   └── ctc_utils.py     # CTC 编码 / 贪心解码
├── data/                # 数据管线
│   ├── synth.py         # 文本行合成生成器（字体/背景/旋转/噪声）
│   ├── synth_det.py     # 整图文本框合成生成器
│   └── td500.py         # MSRA-TD500 解析 → DBNet GT 掩膜
└── examples/            # 训练 / 推理脚本（都支持断点续训）
    ├── pretrain_ctc.py          # 识别预训练
    ├── pretrain_detector.py     # 检测预训练 / 微调
    ├── infer_ctc.py             # 识别单图推理 demo
    └── train_recognizer.py      # 识别训练（EOS 并行解码对比版）
```

## 训练结果

| 任务 | 数据 | 结果 |
|---|---|---|
| 文本识别（10 数字，全增强） | 合成 3000 张 | **整串 98.5%** |
| 文本识别（36 字母数字，全增强） | 合成 5000 张 | **整串 94.5%** |
| 文本检测（合成域内） | 合成整图 | **二值掩膜 IoU 94.3%** |
| 文本检测（真实数据） | MSRA-TD500（含 SynthText 预训练中）| 见 Roadmap |

## 快速开始

```bash
# 识别预训练（合成数据）
PYTHONPATH="E:/Sam2/ocr" python examples/pretrain_ctc.py \
    --alphabet "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" \
    --fixed_size 5000 --steps 4000 --ckpt checkpoints/ctc_full.pt

# 识别单图推理（加载训练好的权重）
PYTHONPATH="E:/Sam2/ocr" python examples/infer_ctc.py --ckpt checkpoints/ctc_full.pt

# 检测预训练（合成整图）
PYTHONPATH="E:/Sam2/ocr" python examples/pretrain_detector.py --data synth \
    --steps 2000 --ckpt checkpoints/det_synth.pt

# 检测微调（真实数据）
PYTHONPATH="E:/Sam2/ocr" python examples/pretrain_detector.py \
    --data <MSRA-TD500路径> --lr 1e-4 --init_ckpt checkpoints/det_synth.pt \
    --steps 1500 --ckpt checkpoints/det_td500.pt
```

所有训练脚本均支持 `--resume` 断点续训、日志落盘、验证集评估。

## 数据集

| 数据集 | 用途 | 说明 |
|---|---|---|
| 合成文本行 | 识别预训练 | 自研生成器，多字体/背景/旋转/噪声，标签精确 |
| 合成整图 | 检测预训练 | 自研生成器，随机背景 + 文本框 |
| MSRA-TD500 | 检测真实微调 | 500 张自然场景，旋转框标注 |
| SynthText | 检测预训练（进行中）| 80 万张自然背景合成图（约 40GB）|

## Roadmap

- [x] 识别头（CTC）合成预训练，36 字符 94.5%
- [x] 检测头（DBNet）+ MSRA-TD500 数据管线
- [x] 合成检测预训练（域内 IoU 94%）
- [ ] SynthText 检测预训练 → TD500 微调（数据下载中）
- [ ] 检测框 → 识别头 端到端串联
- [ ] 中文词表扩展 / 模型轻量化

---

*Acknowledgments: 视觉特征部分的设计参考了 Hiera（层次化视觉 Transformer）及 FPN 等公开发表的架构思路；检测/识别头、数据管线、训练基础设施为本项目独立实现。*
