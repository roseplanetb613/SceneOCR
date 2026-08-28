"""CTC 识别头：把文本行图变成一列列特征，用 CTC 对齐到字符序列。

为什么用 CTC（治并行解码器的病）:
    并行解码器(RecognitionHead)靠"固定位置的查询 token"去图里对齐字符,
    一旦字符位置随机/旋转就崩。CTC 换了个思路:
      1. 图像 → 一列列特征 (时间步 T, 每列一个向量);
      2. 每列在 [词表 + blank] 上独立打分;
      3. ctc_loss 自动对齐"哪列对应哪个字符", blank 处理重复字和长短不一。
    模型不用猜对齐, 只需让每一列的特征足够区分"是哪个字符 / 是空白"。

积木仍全部来自 components/:
    stem      →  Conv2d + CXBlock (blocks)
    序列编码  →  Attention + MLP (attention) —— 让相邻列互相看, 增强序列上下文
    输出头    →  Linear
"""

import torch
import torch.nn as nn

from components.blocks import CXBlock
from components.attention import Attention, MLP


class CTCEncoderLayer(nn.Module):
    """标准的自注意力编码层: 自注意力 + 前馈 (pre-norm)。"""

    def __init__(self, d_model, nhead, dim_ff):
        super().__init__()
        self.attn = Attention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dim_ff, d_model, 2, activation=nn.GELU)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(x, x, x)   # (B, T, d_model)
        x = self.norm1(x)
        x = x + self.mlp(x)
        x = self.norm2(x)
        return x


class CTCHead(nn.Module):
    def __init__(self, d_model=128, vocab_size=11, img_h=32, num_encoder_layers=2):
        """
        vocab_size: 字母表大小(不含 blank)。输出通道 = vocab_size + 1 (blank 在最后)。
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.blank_idx = vocab_size  # blank = 最后一个索引

        # ---- 骨干: conv stem + CXBlock → (B, d_model, H', W') ----
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GELU(),
            CXBlock(dim=64),
            nn.Conv2d(64, d_model, 3, stride=2, padding=1),
            nn.GELU(),
            CXBlock(dim=d_model),
        )

        # ---- 序列编码: 相邻列互相看, 增强上下文 ----
        self.encoder = nn.Sequential(
            *[CTCEncoderLayer(d_model, nhead=4, dim_ff=d_model * 4)
              for _ in range(num_encoder_layers)]
        )

        # ---- 输出: 每列 → (vocab+1) 打分 ----
        self.head = nn.Linear(d_model, vocab_size + 1)

    def forward(self, x):
        """
        x: (B, 3, H, W) 文本行图
        返回 logits: (B, T, vocab+1), 其中 T = W' = 特征列数
        """
        feats = self.stem(x)          # (B, d_model, H', W')
        seq = feats.mean(dim=2)       # 高度方向平均 → (B, d_model, W') 一列一向量
        seq = seq.transpose(1, 2)     # (B, W', d_model)
        seq = self.encoder(seq)       # 序列建模
        logits = self.head(seq)       # (B, T, vocab+1)
        return logits
