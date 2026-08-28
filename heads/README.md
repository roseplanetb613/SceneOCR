# heads/ —— OCR 双头

## 检测头（文本定位）

- `detection.py` — **DBNetHead**：概率图 + 阈值图 + 可微分二值化
  - 输入：FPN 多尺度特征（BCHW）
  - 输出：prob / thr / binary 三张图（stride-4 分辨率）
- `losses.py` — DBNet 损失：`L_s(概率图 OHEM BCE+Dice) + αL_t(阈值图 L1) + βL_b(二值图 BCE)`
  - **OHEM 难例挖掘**：解决文本像素占比过小导致的类别不平衡
- `postprocess.py` — 概率图 → 文本框（阈值二值化 + 连通域 + 外接框）

## 识别头（文本读取）

- `recognition_ctc.py` — **CTCHead**：卷积骨干 → 序列编码 → 逐列分类（CTC 对齐）
  - 不依赖固定位置对齐，对变长文本/随机偏移/旋转天然健壮
- `ctc_utils.py` — CTC 编码 / 贪心解码
- `recognition.py` — 并行解码版识别头（EOS 查询式，供架构对比）
- `recognition_utils.py` — 词表 / EOS 编解码工具
