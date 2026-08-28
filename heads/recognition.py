"""识别头：复用 components/attention/ 的 TwoWayTransformer 做解码，把文本行图读成字符序列。

设计（查询式并行解码，非自回归、非 CTC）:
    文本行图 → 骨干(conv stem + CXBlock) → 2D 特征 (B, C, H', W')
             → 位置编码 (PositionEmbeddingSine)
             → 一组可学习的"字符查询 token"(每位置一个, 共 max_len 个)
             → TwoWayTransformer: 查询 token 和视觉特征双向注意力
             → 每个查询 token → Linear → 字符 logits (B, max_len, vocab)

所有积木都来自 components/:
    CXBlock            → blocks.convnext
    PositionEmbeddingSine → neck.position_encoding
    TwoWayTransformer  → attention.transformer
    输出 MLP/Linear     → attention.mlp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.blocks import CXBlock
from components.neck import PositionEmbeddingSine
from components.attention import TwoWayTransformer


class RecognitionHead(nn.Module):
    def __init__(
        self,
        d_model: int = 128,          # 特征/解码通道
        vocab_size: int = 11,        # 字符表大小(含 pad)
        max_len: int = 4,            # 最大字符长度 = 查询 token 数
        decoder_depth: int = 2,      # TwoWayTransformer 层数
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # ---- 骨干: conv stem + CXBlock（从 components 复用）----
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GELU(),
            CXBlock(dim=64),                    # (B,64,H,W)
            nn.Conv2d(64, d_model, 3, stride=2, padding=1),  # 降采样省计算
            nn.GELU(),
            CXBlock(dim=d_model),               # (B,d_model,H/2,W/2)
        )

        # ---- 位置编码（2D 正弦，components 复用）----
        # 注意: PositionEmbeddingSine 输出通道 = num_pos_feats, 要和 d_model 一致
        self.pos_enc = PositionEmbeddingSine(num_pos_feats=d_model)

        # ---- 解码器：TwoWayTransformer（components 复用）----
        self.decoder = TwoWayTransformer(
            depth=decoder_depth,
            embedding_dim=d_model,
            num_heads=4,
            mlp_dim=d_model * 4,
        )

        # ---- 字符查询 token：每位置一个可学习向量 ----
        self.queries = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

        # ---- 分类头：查询 token → 字符 logits ----
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, H, W) 裁剪好的文本行图（固定高度, 例 32×128）
        返回: logits (B, max_len, vocab)
        """
        B = x.shape[0]
        feats = self.stem(x)        # (B, d_model, H', W')
        pe = self.pos_enc(feats)    # (B, d_model, H', W')

        # 查询 token 扩到 batch
        q = self.queries[None].expand(B, -1, -1)  # (B, max_len, d_model)

        # 双向注意力：查询 <-> 视觉特征
        queries, _ = self.decoder(feats, pe, q)   # (B, max_len, d_model)

        logits = self.head(queries)  # (B, max_len, vocab)
        return logits
