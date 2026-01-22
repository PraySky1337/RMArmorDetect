"""ShuffleNetV2 modules for armor_detect."""

from .backbone import ShuffleNetV2Backbone
from .blocks import (
    ConvBNAct,
    Hardswish,
    ShuffleV2Block,
    ShuffleV2Stage,
    channel_shuffle,
    hardswish,
    make_divisible,
)

__all__ = [
    "ConvBNAct",
    "Hardswish",
    "ShuffleNetV2Backbone",
    "ShuffleV2Block",
    "ShuffleV2Stage",
    "channel_shuffle",
    "hardswish",
    "make_divisible",
]
