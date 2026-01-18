"""
ArmorPoseModel - 独立于 ultralytics 的三分支姿态模型。

该模块提供 ArmorPoseModel 类，用于装甲板检测训练。
与 ultralytics.PoseModel 不同，这个类完全在 armor_detect 中实现，
不会产生 ultralytics → armor_detect 的反向依赖。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ultralytics.nn.tasks as tasks_module
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils import LOGGER

from armor_detect.losses.armor_pose_loss import ArmorPoseLoss
from armor_detect.models.heads.armor_pose_head import ArmorPoseHead
from armor_detect.modules.mobilenetv4.backbone import MobileNetV4Backbone
from armor_detect.modules.shufflenetv2.backbone import ShuffleNetV2Backbone


class ArmorPoseModel(DetectionModel):
    """装甲板检测模型，使用三分支分类 (num + color + size)。

    该类继承自 DetectionModel，并：
    1. 在 init_criterion() 中直接返回 ArmorPoseLoss（无反向依赖）
    2. 在模型构建时临时注入自定义 Pose/Backbone 供 parse_model 解析

    Attributes:
        kpt_shape (tuple): 关键点形状 (num_keypoints, dimensions)。

    Example:
        >>> from armor_detect.models import ArmorPoseModel
        >>> model = ArmorPoseModel("armor_yolo11.yaml", data_kpt_shape=(4, 2))
        >>> trainer.model = model
    """

    def __init__(
        self,
        cfg: str | Path | dict[str, Any] = "armor_yolo11.yaml",
        ch: int = 3,
        nc: int | None = None,
        data_kpt_shape: tuple[int | None, int | None] = (None, None),
        verbose: bool = True,
    ):
        """初始化 ArmorPoseModel。

        Args:
            cfg: 模型配置文件路径或字典。
            ch: 输入通道数。
            nc: 类别数（如果提供，覆盖配置中的值）。
            data_kpt_shape: 关键点形状 (num_keypoints, dimensions)。
            verbose: 是否打印模型信息。
        """
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)

        # 如果提供了 data_kpt_shape，覆盖配置中的值
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg.get("kpt_shape", [])):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg.get('kpt_shape')} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape

        overrides = {
            "Pose": ArmorPoseHead,
            "MobileNetV4Backbone": MobileNetV4Backbone,
            "ShuffleNetV2Backbone": ShuffleNetV2Backbone,
        }
        originals: dict[str, object | None] = {}
        for name, obj in overrides.items():
            originals[name] = getattr(tasks_module, name, None)
            setattr(tasks_module, name, obj)
        try:
            super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        finally:
            for name, original in originals.items():
                if original is None:
                    if hasattr(tasks_module, name):
                        delattr(tasks_module, name)
                else:
                    setattr(tasks_module, name, original)

    def init_criterion(self):
        """返回 ArmorPoseLoss 作为损失函数。

        与 ultralytics.PoseModel 不同，这里直接在 armor_detect 中
        返回 ArmorPoseLoss，避免了反向依赖。

        Returns:
            ArmorPoseLoss: 三分支损失函数实例。
        """
        return ArmorPoseLoss(self)


# 向后兼容的别名
PoseModel = ArmorPoseModel
