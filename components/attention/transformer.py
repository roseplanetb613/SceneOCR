"""TwoWayTransformer：SAM 掩膜解码器的 Transformer 主干（双向注意力）。

自实现的双向注意力 Transformer 解码器（TwoWayTransformer）。

普通 Transformer 解码器通常是"query 去 attend 图像"。TwoWayTransformer 的"双向"在于:
    - 稀疏 token(点/框/掩膜 token) → 去 attend 图像特征（token→image）
    - 图像特征 → 反向 attend 稀疏 token（image→token）
    这样两边信息都流动起来, 解码器既知道"提示在哪", 图像也知道"该看哪"。

层结构 (TwoWayAttentionBlock):
    self-attn(token 内部) → cross-attn(token→image) → MLP → cross-attn(image→token)
"""

from typing import Tuple, Type

import torch
from torch import nn, Tensor

from components.attention.mlp import MLP
from components.attention.attention import Attention


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,  # 层数, 例 2
        embedding_dim: int,  # 通道, 例 256
        num_heads: int,  # 头数, 例 8
        mlp_dim: int,  # MLP 隐藏宽, 例 2048
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,  # 注意力内部降维, 省显存
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),  # 第一层跳过位置编码(点还没结合提示)
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,  # (B, C, H, W) 图像特征
        image_pe: Tensor,  # (B, C, H, W) 图像位置编码(常为 (1, C, H, W))
        point_embedding: Tensor,  # (B, N, C) 稀疏提示(点/框) token
    ) -> Tuple[Tensor, Tensor]:
        """Returns: 处理后的 point_embedding(查询), 处理后的 image_embedding(键)"""
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        # Prepare queries
        queries = point_embedding  # (B, N_prompt, C): 稀疏提示作为查询
        keys = image_embedding  # (B, HW, C): 图像特征作为键/值

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,  # 注意: 位置编码是"最初的"点嵌入
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out  # 残差
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)  # token 内部自注意力
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # ===== 1. 稀疏 token 自注意力 =====
        # queries (B, N_prompt, C), keys (B, HW, C)
        if self.skip_first_layer_pe:
            # 第一层先不加位置编码(查询还没确定语义)
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out  # 残差
        queries = self.norm1(queries)

        # ===== 2. 稀疏 token → 图像 跨注意力 =====
        # 提示去"看"图像特征, 把图像信息吸收进提示 token
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # ===== 3. MLP =====
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # ===== 4. 图像 → 稀疏 token 跨注意力（这就是"双向"的另一半）=====
        # 图像反过来 attend 提示, 让图像特征也知道提示在哪
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        # 注意: 这里 q/k 对调了 —— 图像特征(keys)作 query, 提示(queries)作 key/value
        keys = keys + attn_out  # 更新的是图像特征
        keys = self.norm4(keys)

        return queries, keys
