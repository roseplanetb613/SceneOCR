import torch
import torch.nn as nn
from functools import partial
from typing import Optional, Type, Union, Tuple


class Mlp(nn.Module):
    """
    可配置层数的 MLP。

    num_layers=2:
        in → hidden → out

    num_layers=3:
        in → hidden → hidden → out

    num_layers=4:
        in → hidden → hidden → hidden → out

    use_conv=True:
        使用 1x1 Conv2d，否则使用 Linear。
    """

    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            num_layers: int = 2,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Optional[Type[nn.Module]] = None,
            bias: Union[bool, Tuple[bool, bool]] = True,
            drop: Union[float, Tuple[float, ...]] = 0.,
            use_conv: bool = False,
            device=None,
            dtype=None,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers 必须 >= 1")

        dd = {
            "device": device,
            "dtype": dtype,
        }

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # bias:
        # True -> (True, True)
        # False -> (False, False)
        if isinstance(bias, bool):
            bias_list = [bias] * num_layers
        else:
            # 如果只给两个值：
            # 第一个用于第一层
            # 第二个用于最后一层
            if len(bias) != 2:
                raise ValueError(
                    "bias 必须是 bool 或长度为 2 的 tuple"
                )

            bias_list = [bias[0]] * (num_layers - 1) + [bias[1]]

        # dropout
        if isinstance(drop, (int, float)):
            drop_probs = [float(drop)] * num_layers
        else:
            if len(drop) == 1:
                drop_probs = list(drop) * num_layers
            elif len(drop) == num_layers:
                drop_probs = list(drop)
            else:
                raise ValueError(
                    "drop 必须是 float、长度为 1 或长度等于 num_layers 的 tuple"
                )

        # Linear 或 1x1 Conv
        linear_layer = (
            partial(nn.Conv2d, kernel_size=1)
            if use_conv
            else nn.Linear
        )

        # 构造每一层的维度
        if num_layers == 1:
            dims = [in_features, out_features]
        else:
            dims = (
                [in_features]
                + [hidden_features] * (num_layers - 1)
                + [out_features]
            )

        self.layers = nn.ModuleList()

        for i in range(num_layers):

            layer = nn.ModuleDict({
                "fc": linear_layer(
                    dims[i],
                    dims[i + 1],
                    bias=bias_list[i],
                    **dd,
                ),

                "act": (
                    act_layer()
                    if i < num_layers - 1
                    else nn.Identity()
                ),

                "drop": nn.Dropout(drop_probs[i]),

                "norm": (
                    norm_layer(
                        dims[i + 1],
                        **dd
                    )
                    if norm_layer is not None and i < num_layers - 1
                    else nn.Identity()
                ),
            })

            self.layers.append(layer)

    def forward(self, x):

        for layer in self.layers:
            x = layer["fc"](x)
            x = layer["act"](x)
            x = layer["drop"](x)
            x = layer["norm"](x)

        return x