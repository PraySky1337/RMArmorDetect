"""
Armor Validator - Validation class for armor detection models.

This module provides the ArmorValidator class for evaluating armor detection models
with dual classification branches (number + color).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import OKS_SIGMA, PoseMetrics, kpt_iou

from armor_detect.utils import NUMBER_CLASSES, COLOR_CLASSES


class ArmorValidator(DetectionValidator):
    """Validator for armor detection with dual classification branches.

    This validator handles pose estimation without bounding boxes, using keypoints
    for both localization and IoU computation. It supports dual classification
    branches for number (1-8) and color (B, R, N, P) prediction.

    Attributes:
        sigma (np.ndarray): Sigma values for OKS calculation.
        kpt_shape (list[int]): Shape of keypoints [nkpt, ndim].
        nc_num (int): Number of number classes.
        nc_color (int): Number of color classes.
        color_names (list[str]): Names of color classes.
        color_correct (int): Count of correct color predictions.
        color_total (int): Total color predictions.
    """

    def __init__(
        self,
        dataloader=None,
        save_dir=None,
        args=None,
        _callbacks=None,
    ) -> None:
        """Initialize ArmorValidator for keypoint-only pose estimation.

        Args:
            dataloader: Dataloader for validation data.
            save_dir: Directory to save validation results.
            args: Configuration arguments.
            _callbacks: Callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.nc_num = 8
        self.nc_color = 4
        self.color_names = COLOR_CLASSES
        self.args.task = "pose"
        self.metrics = PoseMetrics()

        # Color classification statistics
        self.color_correct = 0
        self.color_total = 0

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch by converting keypoints and color data to float.

        Args:
            batch: Input batch dictionary.

        Returns:
            Preprocessed batch dictionary.
        """
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].float()
        if "color" in batch:
            batch["color"] = batch["color"].to(self.device)
        return batch

    def get_desc(self) -> str:
        """Return description of evaluation metrics.

        Returns:
            Formatted string with metric column headers.
        """
        return ("%22s" + "%11s" * 6) % (
            "Class",
            "Images",
            "Instances",
            "P",
            "R",
            "mAP50",
            "mAP50-95",
        )

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize evaluation metrics for pose validation.

        Args:
            model: The model being validated.
        """
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt

        # Get nc_num from model head
        pose_head = self._find_pose_head(model)
        if pose_head:
            self.nc_num = pose_head.nc_num
        else:
            self.nc_num = self.data.get("nc", 8)

        self.nc_color = len(self.data.get("color_names", COLOR_CLASSES))
        self.color_names = self.data.get("color_names", COLOR_CLASSES)
        self.color_correct = 0
        self.color_total = 0

        # Override names from data config
        if "names" in self.data:
            self.names = self.data["names"]
            self.nc = len(self.names)

    def _find_pose_head(self, model: torch.nn.Module):
        """Find the pose head in the model.

        Args:
            model: The model to search.

        Returns:
            The pose head layer if found, None otherwise.
        """
        inner_model = model.model if hasattr(model, "model") else model
        model_layers = inner_model.model if hasattr(inner_model, "model") else inner_model

        for layer in reversed(list(model_layers)):
            if hasattr(layer, "nc_num"):
                return layer
        return None

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Postprocess predictions to extract keypoints and classifications.

        Args:
            preds: Model predictions tensor.

        Returns:
            List of dictionaries containing processed predictions.
        """
        # Handle validation mode output format
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        bs, channels, na = preds.shape
        nc_num = self.nc_num
        nc_color = self.nc_color
        nkpt, ndim = self.kpt_shape
        nk = nkpt * ndim

        results = []
        for b_idx in range(bs):
            pred_data = preds[b_idx]

            # Split predictions
            num_scores = pred_data[:nc_num]
            color_scores = pred_data[nc_num : nc_num + nc_color]
            kpt_data = pred_data[nc_num + nc_color :]

            # Get confidence from number classification
            num_conf, cls_indices = num_scores.max(dim=0)
            color_indices = color_scores.argmax(dim=0)

            # Confidence threshold filtering
            conf_thres = 0.4
            max_dets = 50
            valid_mask = num_conf > conf_thres

            if valid_mask.sum() == 0:
                results.append(self._empty_result(preds.device, nkpt, ndim))
                continue

            # Extract valid predictions
            valid_conf = num_conf[valid_mask]
            valid_cls = cls_indices[valid_mask]
            valid_color = color_indices[valid_mask]
            valid_kpts = kpt_data[:, valid_mask].t().view(-1, nkpt, ndim)

            # Apply keypoint NMS
            keep = self._kpt_nms(valid_kpts, valid_conf, iou_thres=0.5)

            if len(keep) > max_dets:
                keep = keep[:max_dets]

            results.append({
                "cls": valid_cls[keep],
                "conf": valid_conf[keep],
                "keypoints": valid_kpts[keep],
                "color_cls": valid_color[keep],
            })

        return results

    def _empty_result(
        self, device: torch.device, nkpt: int, ndim: int
    ) -> dict[str, torch.Tensor]:
        """Create empty result dictionary.

        Args:
            device: Torch device.
            nkpt: Number of keypoints.
            ndim: Number of dimensions per keypoint.

        Returns:
            Dictionary with empty tensors.
        """
        return {
            "cls": torch.zeros((0,), dtype=torch.long, device=device),
            "conf": torch.zeros((0,), device=device),
            "keypoints": torch.zeros((0, nkpt, ndim), device=device),
            "color_cls": torch.zeros((0,), dtype=torch.long, device=device),
        }

    def _kpt_nms(
        self,
        keypoints: torch.Tensor,
        scores: torch.Tensor,
        iou_thres: float = 0.5,
    ) -> torch.Tensor:
        """Apply NMS based on keypoint IoU.

        Args:
            keypoints: Keypoint predictions (N, nkpt, ndim).
            scores: Confidence scores (N,).
            iou_thres: IoU threshold for NMS.

        Returns:
            Indices of kept predictions.
        """
        if len(keypoints) == 0:
            return torch.tensor([], dtype=torch.long, device=keypoints.device)

        # Sort by confidence
        order = scores.argsort(descending=True)
        keep = []

        while len(order) > 0:
            i = order[0].item()
            keep.append(i)

            if len(order) == 1:
                break

            # Compute keypoint IoU with remaining
            remaining = order[1:]
            kpts_i = keypoints[i : i + 1]
            kpts_rest = keypoints[remaining]

            ious = kpt_iou(kpts_i, kpts_rest, self.sigma)

            # Keep predictions with IoU below threshold
            mask = ious.squeeze(0) < iou_thres
            order = remaining[mask]

        return torch.tensor(keep, dtype=torch.long, device=keypoints.device)

    def update_metrics(
        self,
        preds: list[dict[str, torch.Tensor]],
        batch: dict[str, Any],
    ) -> None:
        """Update metrics with batch predictions.

        Args:
            preds: Processed predictions.
            batch: Ground truth batch data.
        """
        for si, pred in enumerate(preds):
            # Get ground truth for this sample
            idx = batch["batch_idx"] == si
            cls = batch["cls"][idx].squeeze(-1).int()
            kpts = batch["keypoints"][idx]

            # Update color accuracy if available
            if "color" in batch and "color_cls" in pred:
                gt_color = batch["color"][idx]
                pred_color = pred["color_cls"]

                if len(pred_color) > 0 and len(gt_color) > 0:
                    # Simple accuracy: check if any prediction matches
                    self.color_total += len(gt_color)
                    for pc in pred_color:
                        if pc in gt_color:
                            self.color_correct += 1

            # Call parent update_metrics for standard metrics
            super().update_metrics(preds, batch)

    def finalize_metrics(self) -> None:
        """Finalize and log metrics after validation."""
        super().finalize_metrics()

        # Log color accuracy
        if self.color_total > 0:
            color_acc = self.color_correct / self.color_total * 100
            LOGGER.info(f"Color accuracy: {color_acc:.2f}%")


# Alias for backward compatibility
PoseValidator = ArmorValidator
