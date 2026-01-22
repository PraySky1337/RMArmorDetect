# MobileNetV3 Core Blocks
"""
MobileNetV3 building blocks for efficient neural networks.

Reference:
    - Searching for MobileNetV3 (https://arxiv.org/abs/1905.02244)
    - Uses Hardswish activation for better quantization support
    - Squeeze-and-Excitation (SE) attention for channel-wise feature recalibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """Make channels divisible by divisor for optimal hardware performance.

    According to TUP performance guide, channels should be divisible by 8
    to ensure efficient memory access and computation.

    Args:
        v: Target channel count
        divisor: Divisor (default 8 for efficient computation)
        min_value: Minimum value (default is divisor)

    Returns:
        Divisible channel count
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hardswish(x: torch.Tensor) -> torch.Tensor:
    """Hardswish activation function.

    Hardswish(x) = x * ReLU6(x + 3) / 6

    According to TUP-NN-Train performance guide:
    - Proposed in MobileNetV3 as quantization-friendly version of SiLU
    - ~5% faster than SiLU on NUC10 (13.7ms -> 13ms)
    - Better quantization support than SiLU
    - Minimal accuracy loss compared to SiLU
    """
    return x * torch.clamp(x + 3, 0, 6) / 6


class Hardswish(nn.Module):
    """Hardswish activation module."""

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inplace:
            return x.mul_(torch.clamp(x.add_(3), 0, 6).div_(6))  # type: ignore
        return x * torch.clamp(x + 3, 0, 6) / 6


class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + Hardswish/ReLU."""

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        act: bool = True,
        use_hardswish: bool = True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            c1, c2, kernel_size, stride, padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        if act:
            if use_hardswish:
                self.act = Hardswish(inplace=True)
            else:
                self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation attention module.

    This module adaptively recalibrates channel-wise feature responses
    by explicitly modelling interdependencies between channels.
    """

    def __init__(self, c1: int, rd_ratio: float = 0.25, rd_divisor: int = 8):
        """
        Args:
            c1: Input/output channels
            rd_ratio: Reduction ratio for squeeze
            rd_divisor: Divisor for making channels divisible
        """
        super().__init__()
        rd_channels = make_divisible(int(c1 * rd_ratio), rd_divisor)
        self.fc1 = nn.Conv2d(c1, rd_channels, 1)
        self.fc2 = nn.Conv2d(rd_channels, c1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_se = x.mean((2, 3), keepdim=True)
        x_se = hardswish(self.fc1(x_se))
        x_se = torch.sigmoid(self.fc2(x_se))
        return x * x_se


class InvertedResidual(nn.Module):
    """MobileNetV3 Inverted Residual block.

    Structure:
        1x1 Expand -> DW 3x3/5x5 -> SE (optional) -> 1x1 Project (no activation)

    Args:
        c1: Input channels
        c2: Output channels
        expand_ratio: Expansion ratio for hidden channels
        kernel_size: Kernel size for depthwise conv
        stride: Stride for spatial downsampling
        use_se: Whether to use Squeeze-and-Excitation
        use_hs: Whether to use Hardswish (if False, uses ReLU)
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        expand_ratio: float,
        kernel_size: int = 3,
        stride: int = 1,
        use_se: bool = False,
        use_hs: bool = True,
    ):
        super().__init__()
        self.use_shortcut = stride == 1 and c1 == c2

        hidden_ch = make_divisible(int(c1 * expand_ratio))

        layers = []

        # Expand 1x1
        if expand_ratio != 1:
            layers.append(ConvBNAct(c1, hidden_ch, 1, 1, act=use_hs, use_hardswish=use_hs))

        # Depthwise
        layers.append(
            ConvBNAct(
                hidden_ch,
                hidden_ch,
                kernel_size,
                stride,
                groups=hidden_ch,
                act=use_hs,
                use_hardswish=use_hs,
            )
        )

        # Squeeze-and-Excitation
        if use_se:
            layers.append(SqueezeExcite(hidden_ch))

        # Project 1x1 (no activation)
        layers.append(
            nn.Sequential(
                nn.Conv2d(hidden_ch, c2, 1, bias=False),
                nn.BatchNorm2d(c2),
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_shortcut:
            return x + self.block(x)
        return self.block(x)
