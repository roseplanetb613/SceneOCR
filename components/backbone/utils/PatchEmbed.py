from typing import Tuple
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """
        Image2Patch Embedding.
    """
    def __init__(
        self,
        kernel_size: Tuple[int, ...] = (7, 7),  # 卷积核
        stride: Tuple[int, ...] = (4, 4),       # 跳数
        padding: Tuple[int, ...] = (3, 3),      # 补充
        in_chans: int = 3,                      # 3通道rgb
        embed_dim: int = 768,                   # emb维度
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self,x:Tensor) -> Tensor:
        # 输入 x: (B, 3, H, W)，例 (1, 3, 224, 224)
        x = self.proj(x)
        # 7×7 stride-4 padding-3 卷积 → (B, embed_dim, H/4, W/4)，例 (1, 96, 56, 56)
        # 等价于把图像切成 56×56 个 4×4 的 patch，每个 patch 提成 96 维向量
        # 交换维度 B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        # 例 (1, 56, 56, 96) —— 之后所有 Transformer block 都用这种 BHWC 布局
        return x

def window_partition(x: Tensor, window_size:int):
    """
        切窗口大小
        把 B, H, W, C 切成多个小窗口
        例: x (1, 28, 28, 192), window_size=4
            -> 28 能整除 4, 不用 padding
            -> 拆成 (1, 7, 4, 7, 4, 192), 调轴后展平 -> windows (49, 4, 4, 192)
               (49 = 7×7 个窗口, 每个窗口 4×4×192)
        若不能整除会先补零到 Hp/Wp(例 14×14 窗口 8 → 补到 16×16), pad_hw 返回补完的尺寸供反分区裁剪。
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
        window_partition 的逆向操作
        例: windows (49, 2, 2, 384), ws=2, pad_hw=(14,14), hw=(14,14)
            -> B = 49 // (14*14//4) = 1
            -> reshape (1,7,7,2,2,384) -> 拼回 (1,14,14,384)
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]  # 裁掉 partition 时补的零
    return x
