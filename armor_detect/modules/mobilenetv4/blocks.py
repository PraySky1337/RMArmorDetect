# MobileNetV4 Core Blocks
"""
Universal Inverted Bottleneck (UIB) and related modules for MobileNetV4.

Reference:
    MobileNetV4 - Universal Models for the Mobile Ecosystem (arXiv:2404.10518)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


def make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """Make channels divisible by divisor (usually 8 for efficient computation)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + Activation (standard building block)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        act: bool = True,
    ):
        """
        Args:
            c1: Input channels
            c2: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Padding (auto-calculated if None)
            groups: Number of groups for grouped convolution
            act: Whether to use activation (ReLU6)
        """
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            c1, c2, kernel_size, stride, padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU6(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation attention module."""

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
        x_se = F.relu(self.fc1(x_se), inplace=True)
        x_se = torch.sigmoid(self.fc2(x_se))
        return x * x_se


class UIB(nn.Module):
    """Universal Inverted Bottleneck - MobileNetV4 core block.

    This unified block can represent 4 different architectures:
    - ExtraDW: start_dw_kernel > 0 and middle_dw_kernel > 0
    - ConvNext-like: start_dw_kernel > 0 and middle_dw_kernel == 0
    - InvertedBottleneck (IB): start_dw_kernel == 0 and middle_dw_kernel > 0
    - FFN: start_dw_kernel == 0 and middle_dw_kernel == 0

    Structure:
        [Optional StartDW] -> Expand 1x1 -> [Optional MiddleDW] -> Project 1x1

    Args:
        c1: Input channels
        c2: Output channels
        expand_ratio: Expansion ratio for hidden channels
        start_dw_kernel: Kernel size for start depthwise conv (0 to disable)
        middle_dw_kernel: Kernel size for middle depthwise conv (0 to disable)
        stride: Stride for spatial downsampling (applied to middle DW or start DW)
        use_se: Whether to use Squeeze-and-Excitation
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        expand_ratio: float = 4.0,
        start_dw_kernel: int = 0,
        middle_dw_kernel: int = 3,
        stride: int = 1,
        use_se: bool = False,
    ):
        super().__init__()
        self.stride = stride
        self.use_shortcut = stride == 1 and c1 == c2

        hidden_channels = make_divisible(int(c1 * expand_ratio))

        # Optional start depthwise conv (before expansion)
        self.start_dw = None
        if start_dw_kernel > 0:
            # If no middle DW, apply stride here
            dw_stride = stride if middle_dw_kernel == 0 else 1
            self.start_dw = ConvBNAct(
                c1, c1, start_dw_kernel, dw_stride, groups=c1, act=True
            )

        # Expansion 1x1 conv
        self.expand = ConvBNAct(c1, hidden_channels, 1, 1, act=True)

        # Optional middle depthwise conv (after expansion)
        self.middle_dw = None
        if middle_dw_kernel > 0:
            self.middle_dw = ConvBNAct(
                hidden_channels, hidden_channels, middle_dw_kernel, stride,
                groups=hidden_channels, act=True
            )

        # Optional SE attention
        self.se = SqueezeExcite(hidden_channels) if use_se else None

        # Projection 1x1 conv (no activation)
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        # Start depthwise conv
        if self.start_dw is not None:
            x = self.start_dw(x)

        # Expansion
        x = self.expand(x)

        # Middle depthwise conv
        if self.middle_dw is not None:
            x = self.middle_dw(x)

        # SE attention
        if self.se is not None:
            x = self.se(x)

        # Projection
        x = self.project(x)

        # Residual connection
        if self.use_shortcut:
            x = x + shortcut

        return x


class ExtraDW(UIB):
    """ExtraDW variant: Both start and middle DW enabled.

    Best for increasing receptive field efficiently.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        expand_ratio: float = 4.0,
        kernel_size: int = 3,
        stride: int = 1,
        use_se: bool = False,
    ):
        super().__init__(
            c1, c2, expand_ratio,
            start_dw_kernel=kernel_size,
            middle_dw_kernel=kernel_size,
            stride=stride,
            use_se=use_se,
        )


class ConvNextBlock(UIB):
    """ConvNext-like variant: Only start DW enabled.

    Performs spatial mixing before expansion (cheaper with larger kernels).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        expand_ratio: float = 4.0,
        kernel_size: int = 7,
        stride: int = 1,
        use_se: bool = False,
    ):
        super().__init__(
            c1, c2, expand_ratio,
            start_dw_kernel=kernel_size,
            middle_dw_kernel=0,
            stride=stride,
            use_se=use_se,
        )


class InvertedBottleneck(UIB):
    """Classic Inverted Bottleneck (MobileNetV2 style).

    Only middle DW enabled - spatial mixing on expanded features.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        expand_ratio: float = 4.0,
        kernel_size: int = 3,
        stride: int = 1,
        use_se: bool = False,
    ):
        super().__init__(
            c1, c2, expand_ratio,
            start_dw_kernel=0,
            middle_dw_kernel=kernel_size,
            stride=stride,
            use_se=use_se,
        )


class FFN(UIB):
    """FFN variant: Pure pointwise convolutions (ViT-style).

    Most accelerator-friendly but limited receptive field.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        expand_ratio: float = 4.0,
        stride: int = 1,
    ):
        super().__init__(
            c1, c2, expand_ratio,
            start_dw_kernel=0,
            middle_dw_kernel=0,
            stride=stride,
            use_se=False,
        )


# =============================================================================
# YAML-Compatible Module Wrappers
# =============================================================================

class MNV4Block(nn.Module):
    """YAML-compatible MobileNetV4 block.

    This wrapper allows using UIB variants in YAML configurations.

    Args:
        c1: Input channels (auto-injected by parse_model)
        c2: Output channels
        block_type: One of 'uib', 'extra_dw', 'convnext', 'ib', 'ffn'
        expand_ratio: Expansion ratio
        kernel_size: DW kernel size
        stride: Stride for downsampling
        use_se: Whether to use SE attention
    """

    BLOCK_TYPES = {
        'uib': UIB,
        'extra_dw': ExtraDW,
        'convnext': ConvNextBlock,
        'ib': InvertedBottleneck,
        'ffn': FFN,
    }

    def __init__(
        self,
        c1: int,
        c2: int,
        block_type: str = 'ib',
        expand_ratio: float = 4.0,
        kernel_size: int = 3,
        stride: int = 1,
        use_se: bool = False,
    ):
        super().__init__()

        block_cls = self.BLOCK_TYPES.get(block_type, InvertedBottleneck)

        if block_type == 'ffn':
            self.block = block_cls(c1, c2, expand_ratio, stride)
        elif block_type == 'uib':
            # For UIB, kernel_size is used for middle_dw_kernel
            self.block = block_cls(
                c1, c2, expand_ratio,
                start_dw_kernel=0,
                middle_dw_kernel=kernel_size,
                stride=stride,
                use_se=use_se,
            )
        else:
            self.block = block_cls(c1, c2, expand_ratio, kernel_size, stride, use_se)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MNV4Stem(nn.Module):
    """MobileNetV4 Stem - Initial convolution layers.

    Reduces spatial resolution by 4x (two stride-2 convolutions).

    Args:
        c1: Input channels (typically 3 for RGB)
        c2: Output channels
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        c_mid = make_divisible(c2 // 2)

        self.conv1 = ConvBNAct(c1, c_mid, 3, 2)  # /2
        self.conv2 = ConvBNAct(c_mid, c_mid, 3, 1, groups=c_mid)  # DW
        self.conv3 = ConvBNAct(c_mid, c2, 1, 1)  # PW
        self.conv4 = ConvBNAct(c2, c2, 3, 2, groups=c2)  # /2 DW

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


# =============================================================================
# MobileNetV4 Architecture Specifications
# =============================================================================

# Block specification format:
# (block_type, expand_ratio, c_out, kernel_size, stride, use_se)
# block_type: 'ib' = InvertedBottleneck, 'ed' = ExtraDW, 'cn' = ConvNext, 'ffn' = FFN

MNV4_CONV_SMALL_SPEC = {
    # Stage configurations: list of (block_type, expand, c_out, kernel, stride, se)
    'stem_channels': 32,
    'stages': [
        # Stage 1 (P2/4): stride 2 from stem, so already /4
        [
            ('ib', 3.0, 32, 3, 2, False),  # First block with stride
            ('ib', 2.0, 32, 3, 1, False),
        ],
        # Stage 2 (P3/8)
        [
            ('ib', 4.0, 64, 3, 2, False),
            ('ib', 4.0, 64, 3, 1, False),
            ('ib', 3.0, 64, 3, 1, False),
            ('ed', 3.0, 64, 3, 1, False),  # ExtraDW
        ],
        # Stage 3 (P4/16)
        [
            ('ib', 4.0, 96, 3, 2, True),
            ('ed', 4.0, 96, 3, 1, True),
            ('ib', 4.0, 96, 3, 1, True),
            ('ib', 3.0, 96, 3, 1, True),
            ('ed', 4.0, 96, 3, 1, True),
            ('ib', 4.0, 96, 3, 1, True),
        ],
        # Stage 4 (P5/32)
        [
            ('ib', 6.0, 128, 3, 2, True),
            ('ed', 4.0, 128, 3, 1, True),
            ('ib', 6.0, 128, 3, 1, True),
            ('ed', 4.0, 128, 5, 1, True),
            ('ib', 6.0, 128, 3, 1, True),
            ('ed', 4.0, 128, 3, 1, True),
        ],
    ],
}

MNV4_CONV_MEDIUM_SPEC = {
    'stem_channels': 32,
    'stages': [
        # Stage 1 (P2/4)
        [
            ('ib', 4.0, 48, 3, 2, False),
            ('ib', 4.0, 48, 3, 1, False),
        ],
        # Stage 2 (P3/8)
        [
            ('ib', 4.0, 80, 3, 2, False),
            ('ib', 4.0, 80, 3, 1, False),
            ('ib', 4.0, 80, 3, 1, False),
            ('ib', 4.0, 80, 3, 1, False),
        ],
        # Stage 3 (P4/16)
        [
            ('ib', 4.0, 160, 3, 2, True),
            ('ib', 4.0, 160, 3, 1, True),
            ('ib', 4.0, 160, 3, 1, True),
            ('ib', 4.0, 160, 3, 1, True),
            ('ed', 4.0, 160, 3, 1, True),
            ('ib', 4.0, 160, 3, 1, True),
            ('ed', 4.0, 160, 5, 1, True),
            ('ib', 4.0, 160, 3, 1, True),
        ],
        # Stage 4 (P5/32)
        [
            ('ib', 6.0, 256, 5, 2, True),
            ('ed', 4.0, 256, 5, 1, True),
            ('ib', 6.0, 256, 5, 1, True),
            ('ed', 4.0, 256, 5, 1, True),
            ('ib', 6.0, 256, 3, 1, True),
            ('ed', 4.0, 256, 5, 1, True),
            ('ib', 6.0, 256, 3, 1, True),
            ('ed', 4.0, 256, 5, 1, True),
            ('ib', 6.0, 256, 3, 1, True),
            ('ib', 6.0, 256, 3, 1, True),
            ('ib', 6.0, 256, 3, 1, True),
        ],
    ],
}

MNV4_CONV_LARGE_SPEC = {
    'stem_channels': 24,
    'stages': [
        # Stage 1 (P2/4)
        [
            ('ib', 4.0, 48, 3, 2, False),
            ('ib', 4.0, 48, 3, 1, False),
        ],
        # Stage 2 (P3/8)
        [
            ('ib', 4.0, 96, 3, 2, False),
            ('ib', 4.0, 96, 3, 1, False),
            ('ib', 4.0, 96, 3, 1, False),
            ('ib', 4.0, 96, 3, 1, False),
            ('ib', 4.0, 96, 3, 1, False),
            ('ed', 4.0, 96, 3, 1, False),
        ],
        # Stage 3 (P4/16)
        [
            ('ib', 4.0, 192, 3, 2, True),
            ('ib', 4.0, 192, 3, 1, True),
            ('ib', 4.0, 192, 3, 1, True),
            ('ib', 4.0, 192, 3, 1, True),
            ('ib', 4.0, 192, 3, 1, True),
            ('ed', 4.0, 192, 3, 1, True),
            ('ib', 4.0, 192, 3, 1, True),
            ('ed', 4.0, 192, 3, 1, True),
            ('ib', 4.0, 192, 3, 1, True),
            ('ed', 4.0, 192, 5, 1, True),
            ('ib', 4.0, 192, 3, 1, True),
        ],
        # Stage 4 (P5/32)
        [
            ('cn', 4.0, 512, 5, 2, True),  # ConvNext-like
            ('ed', 4.0, 512, 5, 1, True),
            ('ib', 4.0, 512, 5, 1, True),
            ('cn', 4.0, 512, 5, 1, True),
            ('ed', 4.0, 512, 5, 1, True),
            ('ib', 4.0, 512, 5, 1, True),
            ('cn', 4.0, 512, 5, 1, True),
            ('ed', 4.0, 512, 5, 1, True),
            ('ib', 4.0, 512, 5, 1, True),
            ('cn', 4.0, 512, 5, 1, True),
            ('ed', 4.0, 512, 5, 1, True),
            ('ib', 4.0, 512, 5, 1, True),
            ('cn', 4.0, 512, 5, 1, True),
        ],
    ],
}


class MNV4Stage(nn.Module):
    """MobileNetV4 Stage - A sequence of UIB blocks.

    This module creates a complete stage based on the variant specification.

    Args:
        c1: Input channels (auto-injected by parse_model)
        c2: Output channels (used for channel scaling)
        n: Number of additional blocks to repeat (for depth scaling)
        variant: 'conv_small', 'conv_medium', 'conv_large'
        stage_idx: Stage index (1-4), determines which blocks to use
        width_mult: Channel width multiplier
    """

    SPECS = {
        'conv_small': MNV4_CONV_SMALL_SPEC,
        'conv_medium': MNV4_CONV_MEDIUM_SPEC,
        'conv_large': MNV4_CONV_LARGE_SPEC,
    }

    BLOCK_MAP = {
        'ib': InvertedBottleneck,
        'ed': ExtraDW,
        'cn': ConvNextBlock,
        'ffn': FFN,
    }

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 0,
        variant: str = 'conv_small',
        stage_idx: int = 1,
        width_mult: float = 1.0,
    ):
        super().__init__()

        # Get stage specification
        spec = self.SPECS.get(variant, MNV4_CONV_SMALL_SPEC)
        stage_spec = spec['stages'][stage_idx - 1]  # stage_idx is 1-based

        blocks = []
        in_ch = c1

        for i, (block_type, expand, out_ch, kernel, stride, use_se) in enumerate(stage_spec):
            # Apply width multiplier
            out_ch = make_divisible(int(out_ch * width_mult))

            # Only first block has stride > 1
            block_stride = stride if i == 0 else 1

            block_cls = self.BLOCK_MAP.get(block_type, InvertedBottleneck)

            if block_type == 'ffn':
                block = block_cls(in_ch, out_ch, expand, block_stride)
            else:
                block = block_cls(in_ch, out_ch, expand, kernel, block_stride, use_se)

            blocks.append(block)
            in_ch = out_ch

        # Additional repeated blocks (for depth scaling via 'n')
        if n > 0:
            last_spec = stage_spec[-1]
            block_type, expand, out_ch, kernel, _, use_se = last_spec
            out_ch = make_divisible(int(out_ch * width_mult))
            block_cls = self.BLOCK_MAP.get(block_type, InvertedBottleneck)

            for _ in range(n):
                if block_type == 'ffn':
                    block = block_cls(in_ch, out_ch, expand, 1)
                else:
                    block = block_cls(in_ch, out_ch, expand, kernel, 1, use_se)
                blocks.append(block)
                in_ch = out_ch

        self.blocks = nn.Sequential(*blocks)
        self._out_channels = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

    @property
    def out_channels(self) -> int:
        return self._out_channels
