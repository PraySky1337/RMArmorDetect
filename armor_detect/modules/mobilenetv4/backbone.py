"""MobileNetV4 backbone wrapper for YOLO-style feature extraction."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .blocks import MNV4Stage, MNV4Stem, make_divisible


class MobileNetV4Backbone(nn.Module):
    """MobileNetV4 backbone that returns P3/P4/P5 feature maps.

    Args:
        variant: MobileNetV4 spec name ('conv_small', 'conv_medium', 'conv_large').
        width_mult: Width multiplier applied to channel dimensions.
        out_indices: 1-based stage indices to return (default: (1, 2, 3)).
        in_ch: Input channels (default: 3).
    """

    def __init__(
        self,
        variant: str = "conv_small",
        width_mult: float = 1.0,
        out_indices: Sequence[int] = (1, 2, 3),
        in_ch: int = 3,
    ) -> None:
        super().__init__()

        spec = MNV4Stage.SPECS.get(variant)
        if spec is None:
            raise ValueError(f"Unknown MobileNetV4 variant: {variant}")

        self.out_indices = tuple(int(i) for i in out_indices)
        if not self.out_indices:
            raise ValueError("out_indices must contain at least one stage index")

        self._out_index_set = set(self.out_indices)
        max_stage = max(self.out_indices)
        if max_stage > len(spec["stages"]) or min(self.out_indices) < 1:
            raise ValueError(f"out_indices must be within 1..{len(spec['stages'])}")

        stem_channels = make_divisible(int(spec["stem_channels"] * width_mult))
        self.stem = MNV4Stem(in_ch, stem_channels)

        self.stages = nn.ModuleList()
        self._stage_channels: list[int] = []

        in_channels = stem_channels
        for stage_idx in range(1, max_stage + 1):
            # Note: MNV4Stage determines channels from spec, c1 is only used for
            # first block's input channels, c2 is ignored (determined from spec)
            stage = MNV4Stage(
                c1=in_channels,  # Input channels for first block in stage
                c2=0,  # Ignored (output determined from spec)
                n=0,
                variant=variant,
                stage_idx=stage_idx,
                width_mult=width_mult,
            )
            self.stages.append(stage)
            in_channels = stage.out_channels
            self._stage_channels.append(in_channels)

        self._out_channels = [self._stage_channels[i - 1] for i in self.out_indices]

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
