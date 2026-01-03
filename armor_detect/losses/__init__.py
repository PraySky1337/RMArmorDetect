"""Loss functions for armor detection."""

from armor_detect.losses.wing_loss import WingLoss
from armor_detect.losses.armor_pose_loss import ArmorPoseLoss

__all__ = ['WingLoss', 'ArmorPoseLoss']
