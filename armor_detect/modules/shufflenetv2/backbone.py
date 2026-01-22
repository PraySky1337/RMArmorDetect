"""ShuffleNetV2 backbone wrapper for YOLO-style feature extraction."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBNAct, ShuffleV2Stage, make_divisible


class ShuffleNetV2Backbone(nn.Module):
    """ShuffleNetV2 backbone that returns P3/P4/P5 feature maps.

    Optimized according to TUP-NN-Train performance guide:
    - Uses Hardswish activation instead of ReLU for ~5% speed boost
    - Channels made divisible by 8 for efficient memory access
    - MaxPool replaced with stride-2 conv to reduce fine-grained operations

    Args:
        width_mult: Channel width multiplier (0.5, 1.0, 1.5, 2.0).
        in_ch: Input channels (default: 3).
    """

    STAGE_REPEATS = (4, 8, 4)
    # Stem channels changed from 24 to 32 for divisibility by 8 (TUP optimization)
    STAGE_OUT_CHANNELS = {
        0.5: (32, 48, 96, 192, 1024),
        1.0: (32, 116, 232, 464, 1024),
        1.5: (32, 176, 352, 704, 1024),
        2.0: (32, 244, 488, 976, 2048),
    }

    def __init__(self, width_mult: float = 0.5, in_ch: int = 3) -> None:
        super().__init__()
        width_mult = float(width_mult)
        if width_mult not in self.STAGE_OUT_CHANNELS:
            raise ValueError(f"Unsupported width_mult {width_mult}")

        out_channels = self.STAGE_OUT_CHANNELS[width_mult]
        stem_out = make_divisible(out_channels[0])

        # Initial conv: use Hardswish activation
        self.conv1 = ConvBNAct(in_ch, stem_out, kernel_size=3, stride=2, padding=1, use_hardswish=True)

        # Replace MaxPool with stride-2 conv (TUP: avoid fine-grained operations like MaxPool)
        # Note: keeping same channels as stem_out to maintain architecture compatibility
        self.downsample = ConvBNAct(stem_out, stem_out, kernel_size=3, stride=2, padding=1, use_hardswish=True)

        self.stage2 = ShuffleV2Stage(stem_out, out_channels[1], self.STAGE_REPEATS[0])
        self.stage3 = ShuffleV2Stage(out_channels[1], out_channels[2], self.STAGE_REPEATS[1])
        self.stage4 = ShuffleV2Stage(out_channels[2], out_channels[3], self.STAGE_REPEATS[2])

        self._out_channels = [out_channels[1], out_channels[2], out_channels[3]]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.conv1(x)
        x = self.downsample(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return [p3, p4, p5]

    @property
    def out_channels(self) -> list[int]:
        return list(self._out_channels)
