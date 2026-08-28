"""2D 轴向 RoPE（Rotary Positional Encoding）辅助函数。

自实现的旋转位置编码（RoPE）频率计算与施加。
作用：生成"旋转位置编码"的频率表，并把它乘进 q/k，让注意力天然感知相对位置。
理解要点：
  - 二维特征图有 (x, y) 两个轴向，对每个轴向各算一套 1D RoPE 频率，再拼起来 → "axial"。
  - 频率表用复数表示: e^{iθ} = cosθ + i·sinθ（torch.polar 把 (幅值, 角度) 转成复数）。
  - 施加 RoPE = 把 q/k 的每对相邻维度看成一个复数，乘以频率表(旋转)，再拆回实数。
"""

import torch


def init_t_xy(end_x: int, end_y: int):
    # 生成一个 (end_x * end_y, ) 的网格坐标，按行优先排列
    # 例 end_x=4, end_y=3 → 12 个位置, 返回:
    #   t_x = [0,1,2,3, 0,1,2,3, 0,1,2,3]  (列号, 周期 = end_x)
    #   t_y = [0,0,0,0, 1,1,1,1, 2,2,2,2]  (行号)
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    # 计算二维网格上每个位置、每条频率通道的旋转角度，返回复数表 freqs_cis。
    # dim: 每个头的通道数(需能被 4 整除); end_x/end_y: 网格宽高(最大分辨率)。
    # 返回形状 (end_x * end_y, dim/4 * 4)?? 实际是 (end_x*end_y, dim)，见下。

    # 频率基准: 与标准 RoPE/正弦位置编码相同的几何级数衰减
    # 例 dim=256: 频率下标 0..255, 每条 = 10000^( -2k/dim )
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)  # 每个 token 的列号 / 行号
    # 位置 × 频率 → 每个 token 每个频率的旋转角度
    # freqs_x: (dim/4,), t_x: (N,), 外积 → (N, dim/4)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    # 转成复数 e^{iθ}: 幅值 1, 角度 = 计算好的角度
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    # x、y 各占 dim/2 条通道, 拼成 (N, dim)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # 把频率表 (N, C) reshape 成能对 x (B, nHead, N, C) 广播的形状
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    # 例 x.ndim=4 → 形状变成 (1, 1, N, C)
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    # 把 q、k 施加旋转位置编码。核心: 把最后一条维的相邻两维凑成复数, 乘频率(旋转), 再拆回。
    # xq/xk: (B, nHead, N, C), 其中 C 必须是偶数
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    # reshape (B, nHead, N, C) -> (B, nHead, N, C/2, 2) -> view_as_complex -> (B, nHead, N, C/2)
    xk_ = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0  # k 可能为空(极端情况), 此时不处理
        else None
    )
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)  # (1, 1, N, C/2)
    # 复数相乘 = 旋转: xq * e^{iθ}
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    # view_as_real -> (B, nHead, N, C/2, 2) -> flatten -> (B, nHead, N, C)
    if xk_ is None:
        # no keys to rotate, due to dropout
        return xq_out.type_as(xq).to(xq.device), xk
    # repeat freqs along seq_len dim to match k seq_len
    # 跨注意力场景: 记忆池的 key 数可能比 query 数多, 需要把 query 侧的频率沿 key 长度重复
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]  # key 数 / query 数
        if freqs_cis.is_cuda:
            freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
        else:
            # torch.repeat on complex numbers may not be supported on non-CUDA devices
            # (freqs_cis has 4 dims and we repeat on dim 2) so we use expand + flatten
            freqs_cis = freqs_cis.unsqueeze(2).expand(-1, -1, r, -1, -1).flatten(2, 3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)
