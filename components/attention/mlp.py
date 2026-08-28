"""通用 MLP（多层感知机），用作注意力的前馈网络。"""

import torch.nn as nn
import torch.nn.functional as F


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    """简单多层感知机: input_dim → hidden_dim → ... → output_dim（最后一层不加激活）。

    例 MaskDecoder 里的 output_hypernetworks_mlps: 256 → 256 → 256 → 32
    例 iou_prediction_head: 256 → 256 → 4
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)  # 中间层都宽 hidden_dim
        # 维度链: [input_dim] + h -> h + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # 除最后一层外都加激活
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)  # 可选: 输出压到 (0,1)，用于预测 IoU
        return x
