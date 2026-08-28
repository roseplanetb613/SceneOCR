"""
按功能分组:
    backbone/   图像特征提取骨干（Hiera）
    neck/       特征金字塔 FPN + 位置编码
    attention/  注意力/Transformer 块
    blocks/     通用 CNN 块 / 归一化 / DropPath / 工具函数
"""

from . import backbone, neck, attention, blocks

__all__ = ["backbone", "neck", "attention", "blocks"]
