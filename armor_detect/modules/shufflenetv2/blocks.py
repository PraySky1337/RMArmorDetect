"""ShuffleNetV2 building blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


def make_divisible(v: float, divisor: int = 8, min_value: int | None = None) -> int:
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
    """Convolution + BatchNorm + Hardswish.

    Using Hardswish instead of ReLU/SiLU for better performance on edge devices.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        act: bool = True,
        use_hardswish: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(c1, c2, kernel_size, stride, padding, groups=groups, bias=False)
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


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    """Shuffle channels for cross-group information flow."""
    b, c, h, w = x.size()
    if c % groups != 0:
        raise ValueError(f"Channels {c} not divisible by groups {groups}")
    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class ShuffleV2Block(nn.Module):
    """ShuffleNetV2 block with stride 1 or 2."""

    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")

        self.stride = stride
        branch_features = out_ch // 2

        if stride == 1:
            if in_ch != out_ch:
                raise ValueError("in_ch must equal out_ch when stride=1")
            if in_ch % 2 != 0:
                raise ValueError("in_ch must be divisible by 2 when stride=1")

            self.branch2 = nn.Sequential(
                ConvBNAct(branch_features, branch_features, 1, 1),
                ConvBNAct(
                    branch_features,
                    branch_features,
                    3,
                    stride=1,
                    padding=1,
                    groups=branch_features,
                    act=False,
                ),
                ConvBNAct(branch_features, branch_features, 1, 1),
            )
        else:
            self.branch1 = nn.Sequential(
                ConvBNAct(in_ch, in_ch, 3, stride=2, padding=1, groups=in_ch, act=False),
                ConvBNAct(in_ch, branch_features, 1, 1),
            )
            self.branch2 = nn.Sequential(
                ConvBNAct(in_ch, branch_features, 1, 1),
                ConvBNAct(
                    branch_features,
                    branch_features,
                    3,
                    stride=2,
                    padding=1,
                    groups=branch_features,
                    act=False,
                ),
                ConvBNAct(branch_features, branch_features, 1, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            # Split channels into two branches (identity + transform)
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)


class ShuffleV2Stage(nn.Module):
    """A stage of ShuffleNetV2 blocks."""

    def __init__(self, in_ch: int, out_ch: int, repeats: int) -> None:
        super().__init__()
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        blocks = [ShuffleV2Block(in_ch, out_ch, stride=2)]
        for _ in range(repeats - 1):
            blocks.append(ShuffleV2Block(out_ch, out_ch, stride=1))
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
