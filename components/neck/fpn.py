"""FPN Neck（Feature Pyramid Network 变体）：把骨干输出的多尺度特征统一到一个通道数、再逐层融合。

自实现的特征金字塔（FPN）neck 变体，用于多尺度特征融合。

为什么需要 neck：
    Hiera（trunk）输出 4 个尺度、每个尺度通道不同:  (96,56,56) (192,28,28) (384,14,14) (768,7,7)。
    SAM 后面的 attention 层统一用 d_model=256 通道，且需要"每个尺度都带语义信息"。
    FpnNeck 就做两件事:
      1. 用 1×1 卷积把各尺度都压到 d_model=256 通道（lateral connection）;
      2. 从最粗的尺度开始，逐层把高层语义 2× 上采样，加到下层特征上（top-down），
         让浅层特征也能感知"整张图在表达什么"。
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """骨架（trunk）+ 脖子（neck）的组合器，即视觉特征编码器整体。

    forward 返回的 dict 是后续检测/识别模块拿到的 backbone_out:
      - vision_features: 最粗一层特征 (B, 256, 64, 64)（1024 输入时）
      - vision_pos_enc:  每层对应的位置编码
      - backbone_fpn:    全部 FPN 层特征（细→粗）
    注意: 检查点里这一整块的权重前缀就是 image_encoder.trunk.* / image_encoder.neck.*，
    所以组装整机时必须保留这个包装类。
    """

    def __init__(
        self,
        trunk: nn.Module,  # Hiera 骨干
        neck: nn.Module,   # FpnNeck
        scalp: int = 0,    # 丢弃的最后一层数，常用配置里是 1（扔掉最粗层）
    ):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert (
            self.trunk.channel_list == self.neck.backbone_channel_list
        ), f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"
        # trunk.channel_list=[768,384,192,96](粗→细), neck.backbone_channel_list 必须一致

    def forward(self, sample: torch.Tensor):
        # 图像 (B, 3, H, W) → trunk 多尺度特征 → neck 统一通道 + 融合 + 位置编码
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            # 丢弃最粗的层（低分辨率、语义最重，对分割用处不大还费算力）
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]  # 剩的最粗层 = 主特征
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        return output


class FpnNeck(nn.Module):
    """
    A modified variant of Feature Pyramid Network (FPN) neck
    (we remove output conv and also do bicubic interpolation similar to ViT
    pos embed interpolation)
    """

    def __init__(
        self,
        position_encoding: nn.Module,  # 位置编码模块（PositionEmbeddingSine）
        d_model: int,  # 所有层统一的目标通道数，例 256
        backbone_channel_list: List[int],  # 骨干各层通道数【粗→细】，例 [768,384,192,96]
        kernel_size: int = 1,  # 侧连接用 1×1 卷积，只改通道不改分辨率
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",  # 上采样方式; b+ 配置用 nearest
        fuse_type: str = "sum",  # 融合方式 sum 或 avg
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        """Initialize the neck
        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__()
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        self.d_model = d_model
        # 为骨干的每一层建一个 1×1 卷积（lateral connection），把各层通道压到 d_model
        # 例 [768,384,192,96]: convs[0]=768→256, convs[1]=384→256, convs[2]=192→256, convs[3]=96→256
        # 注意顺序: convs[0] 对应【最粗】层（768），因为 forward 里用 convs[n-i] 反着取
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        # levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level.
        # 哪些层要叠加"自上而下"的语义。默认全部叠加。
        # b+ 配置 fpn_top_down_levels=[2,3]: 只给 level 2、3（较粗的两层）做融合,
        # level 0、1（细层）直接用本层 lateral 特征（细层本身已够精细，不需要高层语义? 省计算）。
        if fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):
        # xs: 骨干输出特征，【细→粗】顺序。以用户 Hiera(96基础) + 224 输入为例:
        #   xs[0]=(B,  96, 56, 56)   stage1
        #   xs[1]=(B, 192, 28, 28)   stage2
        #   xs[2]=(B, 384, 14, 14)   stage3
        #   xs[3]=(B, 768,  7,  7)   stage4
        # convs 是【粗→细】建的，所以取用时要反着取: convs[n-i]。

        out = [None] * len(self.convs)  # 各层融合后的特征 (细→粗)
        pos = [None] * len(self.convs)  # 各层对应的位置编码
        assert len(xs) == len(self.convs)
        # fpn forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # forward in top-down order (from low to high resolution)
        # 从【最粗】(低分辨率, 高语义) 到【最细】(高分辨率, 低语义) 迭代 —— FPN 自上而下
        n = len(self.convs) - 1  # 3
        for i in range(n, -1, -1):
            x = xs[i]
            # lateral connection: 1×1 卷积，把这一层压到 d_model 通道
            # 例 i=3: xs[3](B,768,7,7) → convs[0] → (B,256,7,7)
            #      i=0: xs[0](B, 96,56,56) → convs[3] → (B,256,56,56)
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                # 自上而下融合: 把上一层(更粗)的特征 2× 上采样, 加到本层 lateral 上
                # 例 i=2: prev=(B,256,7,7) → 上采样 → (B,256,14,14) 与本层相加
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,  # 每经过一层就翻倍分辨率
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2  # avg 模式: 求和后除以 2，防止数值变大
            else:
                # 不融合（细层 或 没有更粗的层可用）: 直接用本层 lateral
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            # 为该层特征生成位置编码, 供后续注意力使用
            # PositionEmbeddingSine: (B,256,H,W) → (B,256,H,W), 纯坐标公式、无参数
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        # 返回 (细→粗): out = [(B,256,56,56), (B,256,28,28), (B,256,14,14), (B,256,7,7)]
        # pos 与 out 逐层对应
        return out, pos
