import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    """
    SwiGLU:
        y = down_proj(SiLU(gate_proj(x)) * up_proj(x))

    输入:
        x: (..., dim)

    输出:
        y: (..., dim_out)
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        out_dim: int | None = None,
        bias: bool = False,
    ):
        super().__init__()

        if out_dim is None:
            out_dim = dim

        # 门控链路
        self.gate_proj = nn.Linear(
            dim,
            hidden_dim,
            bias=bias,
        )

        # 线性信息链路
        self.up_proj = nn.Linear(
            dim,
            hidden_dim,
            bias=bias,
        )

        # 输出投影
        self.down_proj = nn.Linear(
            hidden_dim,
            out_dim,
            bias=bias,
        )

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # ① 门控链路
        gate = self.act(self.gate_proj(x))

        # ② 线性链路
        up = self.up_proj(x)

        # ③ 门控
        x = gate * up

        # ④ 输出投影
        x = self.down_proj(x)

        return x