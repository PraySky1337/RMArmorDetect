"""
Armor Detect - RoboMaster Armor Detection Module

This module provides armor detection capabilities with:
- Dual classification branches (number + color)
- 4-keypoint localization
- WingLoss for precise keypoint regression
- YOLO11 backbone (C3k2 + SPPF)
"""

__version__ = '1.0.0'

__all__ = [
    'WingLoss',
    'ArmorPoseLoss',
    'ArmorPoseHead',
    'ArmorDataset',
    'ArmorTrainer',
    'ArmorValidator',
]


def __getattr__(name: str):
    """Lazy import to avoid circular dependencies and missing torch."""
    if name == 'WingLoss':
        from armor_detect.losses.wing_loss import WingLoss
        return WingLoss
    elif name == 'ArmorPoseLoss':
        from armor_detect.losses.armor_pose_loss import ArmorPoseLoss
        return ArmorPoseLoss
    elif name == 'ArmorPoseHead':
        from armor_detect.models.heads.armor_pose_head import ArmorPoseHead
        return ArmorPoseHead
    elif name == 'ArmorDataset':
        from armor_detect.data.armor_dataset import ArmorDataset
        return ArmorDataset
    elif name == 'ArmorTrainer':
        from armor_detect.trainers.armor_trainer import ArmorTrainer
        return ArmorTrainer
    elif name == 'ArmorValidator':
        from armor_detect.validators.armor_validator import ArmorValidator
        return ArmorValidator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
