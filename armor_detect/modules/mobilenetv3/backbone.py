"""MobileNetV3 backbone wrapper for YOLO-style feature extraction."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .blocks import ConvBNAct, InvertedResidual, make_divisible


class MobileNetV3Backbone(nn.Module):
    """MobileNetV3 backbone that returns P3/P4/P5 feature maps.

    Optimized according to TUP-NN-Train performance guide:
    - Uses Hardswish activation for ~5% speed boost over SiLU
    - Channels made divisible by 8 for efficient memory access
    - Squeeze-and-Excitation for better feature representation
    - No MaxPool (replaced with stride-2 conv)

    Reference:
        Searching for MobileNetV3 (https://arxiv.org/abs/1905.02244)

    Args:
        variant: 'large' or 'small' variant
        width_mult: Width multiplier applied to channel dimensions
        out_indices: 1-based stage indices to return (default: (2, 4, 7))
        in_ch: Input channels (default: 3)
    """

    # MobileNetV3 Large specification
    # (expand, out_ch, num_blocks, stride, use_se, use_hs)
    LARGE_SPEC = [
        # Stem
        [None, 16, 1, 2, False, False],
        # Stage 1
        [1, 16, 1, 1, False, True],
        # Stage 2 (P2/4)
        [4, 24, 2, 2, False, False],
        # Stage 3 (P3/8)
        [3, 40, 2, 2, True, False],
        [3, 40, 1, 1, True, False],
        [3, 40, 1, 1, True, False],
        # Stage 4 (P4/16)
        [6, 80, 2, 2, False, True],
        [2.5, 80, 1, 1, False, True],
        [2.5, 80, 1, 1, False, True],
        [6, 80, 1, 1, False, True],
        [6, 80, 1, 1, False, True],
        # Stage 5 (P5/32)
        [6, 112, 1, 1, True, True],
        [6, 112, 1, 1, True, True],
        # Stage 6
        [6, 160, 2, 2, True, True],
        [6, 160, 1, 1, True, True],
        [6, 160, 1, 1, True, True],
    ]

    # MobileNetV3 Small specification
    SMALL_SPEC = [
        # Stem
        [None, 16, 1, 2, False, True],
        # Stage 1
        [1, 16, 1, 1, True, True],
        # Stage 2 (P2/4)
        [4.5, 24, 2, 2, False, False],
        # Stage 3 (P3/8)
        [3.67, 24, 1, 1, False, False],
        # Stage 4 (P4/16)
        [4, 40, 2, 2, True, True],
        [6, 40, 1, 1, True, True],
        [6, 40, 1, 1, True, True],
        # Stage 5 (P5/32)
        [3, 48, 2, 2, True, True],
        [3, 48, 1, 1, True, True],
        # Stage 6
        [6, 96, 2, 2, True, True],
        [6, 96, 1, 1, True, True],
    ]

    def __init__(
        self,
        variant: str = "small",
        width_mult: float = 1.0,
        out_indices: Sequence[int] = (3, 5, 9),
        in_ch: int = 3,
    ) -> None:
        super().__init__()

        if variant == "large":
            spec = self.LARGE_SPEC
        elif variant == "small":
            spec = self.SMALL_SPEC
        else:
            raise ValueError(f"Unknown MobileNetV3 variant: {variant}")

        self.out_indices = tuple(int(i) for i in out_indices)
        if not self.out_indices:
            raise ValueError("out_indices must contain at least one stage index")

        self._out_index_set = set(self.out_indices)

        # Build stem
        stem_spec = spec[0]
        stem_c2 = make_divisible(int(stem_spec[1] * width_mult))
        self.stem = ConvBNAct(in_ch, stem_c2, 3, stem_spec[3], act=True, use_hardswish=stem_spec[5])

        # Build stages
        self.stages = nn.ModuleList()
        self._stage_channels: list[int] = []

        in_ch = stem_c2
        for i, s in enumerate(spec[1:], start=1):
            expand, c2, num_blocks, stride, use_se, use_hs = s
            c2 = make_divisible(int(c2 * width_mult))

            stage = self._make_stage(in_ch, c2, expand, num_blocks, stride, use_se, use_hs)
            self.stages.append(stage)
            self._stage_channels.append(c2)
            in_ch = c2

        self._out_channels = [self._stage_channels[i - 1] for i in self.out_indices]

    def _make_stage(
        self,
        in_ch: int,
        out_ch: int,
        expand_ratio: float,
        num_blocks: int,
        stride: int,
        use_se: bool,
        use_hs: bool,
    ) -> nn.Module:
        """Create a MobileNetV3 stage."""
        blocks = []

        # First block with stride
        blocks.append(
            InvertedResidual(
                in_ch, out_ch, expand_ratio, kernel_size=3, stride=stride,
                use_se=use_se, use_hs=use_hs
            )
        )

        # Remaining blocks with stride=1
        for _ in range(num_blocks - 1):
            blocks.append(
                InvertedResidual(
                    out_ch, out_ch, expand_ratio, kernel_size=3, stride=1,
                    use_se=use_se, use_hs=use_hs
                )
            )

        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        outputs: list[torch.Tensor] = []
        for stage_idx, stage in enumerate(self.stages, start=1):
            x = stage(x)
            if stage_idx in self._out_index_set:
                outputs.append(x)
        return outputs

    @property
    def out_channels(self) -> list[int]:
        return list(self._out_channels)
