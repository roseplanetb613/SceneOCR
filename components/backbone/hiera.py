import logging
from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from iopath.common.file_io import g_pathmgr
from timm.layers import DropPath
# ---- 自建参数 ----
from components.backbone.utils import (
    MultiScaleBlock,
    PatchEmbed
)

class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989

    这是一个「多尺度分层 Vision Transformer」骨干网络。

    核心思想：像 CNN 一样逐阶段降低空间分辨率、增大通道数，但中间运算仍是 Transformer 注意力。
    注意力分两种：
      - 窗口注意力 (window attention, window_size > 0)：把特征图切成 ws×ws 的小块，
        只在每个小块内部做自注意力，避免在整张图上算 O(N²) 的注意力，省显存。
      - 全局注意力 (global attention, window_size = 0)：在整张特征图上做自注意力，
        只用在少数几个 block（global_att_blocks=(12,16,20)）。
    第三个关键操作是 Q-pooling：在阶段切换处，只把 Query 用 MaxPool 下采样（Key/Value 不变），
    相当于"下采样 + 注意力同时完成"，还顺便把特征图分辨率降一半。

    默认配置、输入 224×224 时（B=1，C=通道）：
        patch_embed: 224×224×3 → 56×56×96            (stride-4 卷积切 patch, 输出 BHWC)
        stage 1:  blk 0-1   56×56  C=96   窗口 8×8           (2 个 block)
        stage 2:  blk 2     56×56→28×28  C=96→192  窗口 8    (q-pool 切换块)
                  blk 3-4   28×28  C=192  窗口 4×4            (3 个 block)
        stage 3:  blk 5     28×28→14×14  C=192→384 窗口 4    (q-pool 切换块)
                  blk 6-20  14×14  C=384  窗口 14×14，其中 blk 12/16/20 全局注意力 (16 个 block)
        stage 4:  blk 21    14×14→7×7    C=384→768 窗口 14   (q-pool 切换块)
                  blk 22-23 7×7    C=768  窗口 7×7            (2 个 block)

    最终返回 4 个尺度的特征图（BCHW）：
        (B, 96, 56, 56), (B, 192, 28, 28), (B, 384, 14, 14), (B, 768, 7, 7)
    """

    def __init__(
        self,
        embed_dim: int = 96,  # 初始通道数 (patch 嵌入后)
        num_heads: int = 1,   # 初始注意力头数
        drop_path_rate: float = 0.0,  # 随机深度 (stochastic depth)
        q_pool: int = 3,  # 前几个 stage 切换要做 Q-pooling (默认 3 个 stage 都做)
        q_stride: Tuple[int, int] = (2, 2),  # Q-pooling 下采样倍数 = 空间分辨率减半
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # 每 stage 的 block 数: 共 24 个
        dim_mul: float = 2.0,   # stage 切换时通道翻倍
        head_mul: float = 2.0,  # stage 切换时头数翻倍
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # 各 stage 的窗口大小 (非全局注意力块用); 与 stages 一一对应
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # 用全局注意力的 block 编号
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        weights_path=None,
        return_interm_layers=True,  # return feats from every stage
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)  # 总共多少个 block, 例 2+3+16+3 = 24
        self.q_stride = q_stride
        # stage_ends: 每个 stage 的【最后一个 block】的编号。
        # stages=(2,3,16,3), block 编号 0 起:
        #   stage1 是 blk0~1, stage2 是 blk2~4, stage3 是 blk5~20, stage4 是 blk21~23
        #   => stage_ends = [1, 4, 20, 23]
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        # q_pool_blocks: 需要做 Q-pooling(下采样)的 block = 每个新 stage 的【第一个】block。
        # 取前 q_pool 个 stage（最后一个 stage 不切 stage 了, 不参与）。
        # stage_ends[:-1] = [1,4,20], 各自 +1 => [2,5,21]  → 就是 blk2/5/21 三个切换块
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )
        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks  # (12,16,20): 这些 block 窗口置 0 = 全局注意力

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        # 全局位置编码: 基准分辨率 14×14, (1, 96, 14, 14)。实际分辨率不同时用双线性插值放大
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        # 窗口位置编码: 每个 8×8 窗口内共享一份, (1, 96, 8, 8), 平铺到整个特征图
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        # torch.linspace(0, 0.2, 12) 生成一段概率在 0 - 0.2之间的 12 个 均匀分布的概率序列

        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            # 为第 i 个 block 确定三件事: 窗口大小 window_size / 输出通道 dim_out / 是否做 Q-pool
            dim_out = embed_dim
            # 窗口大小是【滞后一个 block】的:
            #   先按当前 stage 的索引取窗口, 然后再判断是否切到下个 stage。
            # 这样 stage 切换块(即 q_pool 块, blk2/5/21)拿到的窗口来自【上一个 stage】,
            # 而它此刻要切的特征图也还是【上一个 stage 的分辨率】→ 正好对得上。
            # 例: blk2 是 stage2 第一个块, 输入仍是 56×56(stage1 分辨率),
            #     于是用 stage1 的窗口 8×8 切分; 真正 28×28 + 窗口 4×4 从 blk3 才开始。
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                # 属于 global_att_blocks 的 block → 窗口置 0 → 不做切窗, 直接全局注意力
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                # i-1 是某 stage 的最后一块 → i 是新 stage 的第一块:
                #   通道翻倍、注意力头数翻倍、stage 前进
                dim_out = int(embed_dim * dim_mul)    # 96 -> 192 -> 384 -> 768
                num_heads = int(num_heads * head_mul) # 1 -> 2 -> 4 -> 8
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,        # 输入通道(上一个 block 留下的)
                dim_out=dim_out,      # 输出通道(跨 stage 时翻倍)
                num_heads=num_heads,
                drop_path=dpr[i],     # 随机深度: 越深的 block 以越高概率整块丢弃
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                # q_stride 非空 = 这是切换块(blk 2/5/21), 会在注意力里做 Q-pooling 下采样
                window_size=window_size,
            )

            embed_dim = dim_out  # 下一块的输入通道 = 本块输出通道
            self.blocks.append(block)

        self.channel_list = (
            # 各 stage 末块(倒序)的输出通道 = 每层特征金字塔的通道数 [768,384,192,96]
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]  # 只要最后一层
        )

        if weights_path is not None:
            with g_pathmgr.open(weights_path, "rb") as f:
                chkpt = torch.load(f, map_location="cpu")
            logging.info("loading Hiera", self.load_state_dict(chkpt, strict=False))

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        # 生成与特征图 (h, w) 等大的位置编码, 返回 BHWC
        h, w = hw
        window_embed = self.pos_embed_window            # (1, 96, 8, 8)
        # 全局位置编码从基准 14×14 双线性插值到 (h, w)
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        # 叠加窗口位置编码: 按倍数平铺到整图, 例 h=w=56 时 tile 因子 (1,1,7,7) -> (1,96,56,56)
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)       # (B, C, H, W) -> (B, H, W, C)
        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # ======== 入口: 输入是一张 BCHW 图像 ========
        # x: (B, 3, H, W)，例 (1, 3, 224, 224)

        x = self.patch_embed(x)
        # 7×7 stride-4 卷积: (B,3,224,224) -> (B,96,56,56) BCHW
        # 再 permute 成 (B,56,56,96) BHWC —— 后面所有 block 内部都用 BHWC,
        # 因为注意力里的 Linear 只作用在最后一维 C 上。
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])
        # x: (B, 56, 56, 96)，叠加位置编码（含全局 + 窗口两部分）

        outputs = []
        for i, blk in enumerate(self.blocks):
            # 每个 block 处理一整张特征图，内部流程（切窗→注意力→拼回→残差→MLP）
            # 的每个形状见 MultiScaleBlock.forward 的注释
            x = blk(x)
            # 一个 stage 的末块结束 或 整网最后一个 block 时，把这层特征转回 BCHW 存下来。
            # 这就是多尺度特征金字塔: stage1 末 96ch → stage2 末 192ch → ...
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                # (B, H, W, C) -> (B, C, H, W)
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs
        # 例: [(B,96,56,56), (B,192,28,28), (B,384,14,14), (B,768,7,7)]

    def get_layer_id(self, layer_name):
        # https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        # 分层学习率衰减(layer decay)的辅助函数: 训练时把每个参数按"深度"归档,
        # 优化器据此对每层参数缩放学习率(分层学习率衰减):
        #   浅层(档位小)学习率极低(几乎保留预训练权重), 深层(档位大)满速学习。
        # 返回档位与缩放系数: 设 num_layers=24, decay=0.7, 则 id 0→0.7^25(≈0), id 24→0.7, id 25→1.0
        num_layers = self.get_num_layers()  # 24

        if layer_name.find("rel_pos") != -1:
            return num_layers + 1  # 相对位置偏置 → 最深层档位(25), 满学习率
        elif layer_name.find("pos_embed") != -1:
            return 0  # 位置编码 → 最浅档位(0), 几乎不动
        elif layer_name.find("patch_embed") != -1:
            return 0  # patch 嵌入卷积 → 最浅档位(0), 几乎不动
        elif layer_name.find("blocks") != -1:
            # 从参数名里抠出 block 编号: "blocks.5.attn.qkv.weight" -> 5 -> 档位 6
            # +1 是因为档位 0 留给了 patch_embed/pos_embed, 所以 block 从 1 开始排,
            # block 0~23 正好铺满档位 1~24, 越靠后的 block 学习率越大
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1  # 其余参数(如最终 norm) → 最深层档位, 满学习率

    def get_num_layers(self) -> int:
        return len(self.blocks)
