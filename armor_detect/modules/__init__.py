# Armor Detect Custom Modules
"""
Custom modules for armor detection.
These modules can be used in YAML configurations.
"""

from .mobilenetv3 import MobileNetV3Backbone
from .mobilenetv4 import (
    ConvBNAct,
    MNV4Block,
    MNV4Stage,
    MNV4Stem,
    MobileNetV4Backbone,
    UIB,
)
from .shufflenetv2 import ShuffleNetV2Backbone

__all__ = (
    "ConvBNAct",
    "MNV4Block",
    "MNV4Stem",
    "MNV4Stage",
    "MobileNetV3Backbone",
    "MobileNetV4Backbone",
    "ShuffleNetV2Backbone",
    "UIB",
)
