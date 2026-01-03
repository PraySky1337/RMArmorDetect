"""
Armor Trainer - Training class for armor detection models.

This module provides the ArmorTrainer class for training armor detection models
with dual classification branches (number + color).
"""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.plotting import plot_images


class ArmorTrainer(yolo.detect.DetectionTrainer):
    """Trainer class for armor detection with dual classification branches.

    This trainer handles pose estimation with keypoint-only detection (no bounding boxes)
    and dual classification (number class + color class).

    Attributes:
        args (dict): Configuration arguments for training.
        model (PoseModel): The pose estimation model being trained.
        data (dict): Dataset configuration including keypoint shape information.
        loss_names (tuple): Names of the loss components (pose, kobj, cls, color).

    Example:
        >>> from armor_detect.trainers import ArmorTrainer
        >>> args = dict(model="armor_yolo11.yaml", data="armor-pose.yaml", epochs=100)
        >>> trainer = ArmorTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks=None,
    ):
        """Initialize ArmorTrainer for training armor detection models.

        Args:
            cfg: Default configuration dictionary containing training parameters.
            overrides: Dictionary of parameter overrides for the default configuration.
            _callbacks: List of callback functions to be executed during training.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> PoseModel:
        """Get pose estimation model with specified configuration and weights.

        Args:
            cfg: Model configuration file path or dictionary.
            weights: Path to the model weights file.
            verbose: Whether to display model information.

        Returns:
            PoseModel: Initialized pose estimation model.
        """
        model = PoseModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose,
        )
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self) -> None:
        """Set keypoints shape attribute and other model attributes."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

        # Set keypoint names
        kpt_names = self.data.get("kpt_names")
        if not kpt_names:
            names = list(map(str, range(self.model.kpt_shape[0])))
            kpt_names = {i: names for i in range(self.model.nc)}
        self.model.kpt_names = kpt_names

    def get_validator(self):
        """Return an instance of ArmorValidator for model evaluation.

        Returns:
            ArmorValidator instance configured for armor detection.
        """
        self.loss_names = ("pose_loss", "kobj_loss", "cls_loss", "color_loss")
        return yolo.pose.PoseValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot training samples with their annotations including color labels.

        For keypoint-only mode, bboxes are excluded from visualization.

        Args:
            batch: Dictionary containing batch data.
            ni: Number of iterations.
        """
        color_names = self.data.get("color_names", None)

        # Create a copy without bboxes for keypoint-only visualization
        plot_batch = {k: v for k, v in batch.items() if k != "bboxes"}

        plot_images(
            labels=plot_batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            names=self.data["names"],
            color_names=color_names,
            on_plot=self.on_plot,
        )

    def get_dataset(self) -> dict[str, Any]:
        """Retrieve the dataset and ensure it contains required keys.

        Returns:
            dict: A dictionary containing dataset configuration.

        Raises:
            KeyError: If required keys are missing from the dataset.
        """
        data = super().get_dataset()
        if "kpt_shape" not in data:
            raise KeyError(
                f"No `kpt_shape` in {self.args.data}. "
                "See https://docs.ultralytics.com/datasets/pose/"
            )
        return data


# Alias for backward compatibility
PoseTrainer = ArmorTrainer
