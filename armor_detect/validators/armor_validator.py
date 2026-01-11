"""
Armor Validator - Validation class for armor detection models.

This module provides the ArmorValidator class for evaluating armor detection models
with triple classification branches (number + color + size).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.models.yolo.pose.val import PoseValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import kpt_iou

from armor_detect.utils import CLASS_NAMES, COLOR_NAMES, SIZE_NAMES


class ArmorValidator(PoseValidator):
    """Validator for armor detection with triple classification branches.

    Extends PoseValidator to add size classification support.
    Handles triple classification branches: number, color, and size.
    """

    def __init__(
        self,
        dataloader=None,
        save_dir=None,
        args=None,
        _callbacks=None,
    ) -> None:
        """Initialize ArmorValidator."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.nc_size = 2
        self.size_names = SIZE_NAMES
        self.size_correct = 0
        self.size_total = 0

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch including size data."""
        batch = super().preprocess(batch)
        if "size" in batch:
            batch["size"] = batch["size"].to(self.device)
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics including size classification."""
        super().init_metrics(model)

        # Get nc_size from model head
        pose_head = self._find_pose_head(model)
        if pose_head:
            self.nc_size = getattr(pose_head, 'nc_size', 2)
        else:
            self.nc_size = self.data.get("nc_size", 2)

        self.size_names = self.data.get("size_names", SIZE_NAMES)
        self.size_correct = 0
        self.size_total = 0

    def _find_pose_head(self, model: torch.nn.Module):
        """Find the pose head in the model."""
        inner_model = model.model if hasattr(model, "model") else model
        model_layers = inner_model.model if hasattr(inner_model, "model") else inner_model

        for layer in reversed(list(model_layers)):
            if hasattr(layer, "nc_num"):
                return layer
        return None

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Postprocess predictions with triple classification branches.

        Overrides parent to handle size branch in addition to num and color.
        """
        # Handle validation mode output format
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        bs, channels, na = preds.shape
        nc_num = self.nc_num
        nc_color = self.nc_color
        nc_size = self.nc_size
        nkpt, ndim = self.kpt_shape
        nk = nkpt * ndim

        results = []
        for b_idx in range(bs):
            pred_data = preds[b_idx]

            # Split predictions: num + color + size + keypoints
            num_scores = pred_data[:nc_num]
            color_scores = pred_data[nc_num : nc_num + nc_color]
            size_scores = pred_data[nc_num + nc_color : nc_num + nc_color + nc_size]
            kpt_data = pred_data[nc_num + nc_color + nc_size :]

            # Get confidence from all classification branches
            num_conf, cls_indices = num_scores.max(dim=0)
            color_conf, color_indices = color_scores.max(dim=0)
            size_conf, size_indices = size_scores.max(dim=0)

            # FIX Bug #4: Use combined confidence (matches inference in preview_pose.py)
            # This aligns training validation with actual deployment performance
            combined_conf = num_conf * color_conf * size_conf

            # Confidence threshold filtering (adjusted for combined conf: 0.5^3 ≈ 0.125)
            conf_thres = getattr(self.args, 'conf', 0.125)
            max_dets = 50
            valid_mask = combined_conf > conf_thres

            if valid_mask.sum() == 0:
                results.append({
                    "cls": torch.zeros((0,), dtype=torch.long, device=preds.device),
                    "conf": torch.zeros((0,), device=preds.device),
                    "keypoints": torch.zeros((0, nkpt, ndim), device=preds.device),
                    "color_cls": torch.zeros((0,), dtype=torch.long, device=preds.device),
                    "size_cls": torch.zeros((0,), dtype=torch.long, device=preds.device),
                })
                continue

            # Extract valid predictions (use combined_conf for confidence output)
            valid_conf = combined_conf[valid_mask]
            valid_cls = cls_indices[valid_mask]
            valid_color = color_indices[valid_mask]
            valid_size = size_indices[valid_mask]
            valid_kpts = kpt_data[:, valid_mask].t().view(-1, nkpt, ndim)

            # Apply keypoint NMS
            iou_thres = getattr(self.args, 'iou', 0.5)
            keep = self._kpt_nms(valid_kpts, valid_conf, iou_thres=iou_thres)

            if len(keep) > max_dets:
                keep = keep[:max_dets]

            results.append({
                "cls": valid_cls[keep],
                "conf": valid_conf[keep],
                "keypoints": valid_kpts[keep],
                "color_cls": valid_color[keep],
                "size_cls": valid_size[keep],
            })

        return results

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare batch with size data."""
        pbatch = super()._prepare_batch(si, batch)

        # Add size data
        idx = batch["batch_idx"] == si
        if "size" in batch:
            pbatch["size"] = batch["size"][idx]
        else:
            pbatch["size"] = torch.zeros((idx.sum(),), dtype=torch.long, device=self.device)

        return pbatch

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Process batch with size classification accuracy."""
        # Call parent for standard keypoint metrics
        stats = super()._process_batch(preds, batch)

        gt_cls = batch["cls"]

        # Size classification accuracy
        if gt_cls.shape[0] > 0 and preds["cls"].shape[0] > 0 and "size" in batch and "size_cls" in preds:
            gt_size = batch["size"]
            pred_size = preds["size_cls"]

            # Match using keypoint IoU
            gt_kpts = batch["keypoints"]
            x_coords = gt_kpts[..., 0]
            y_coords = gt_kpts[..., 1]
            area = (x_coords.max(dim=1)[0] - x_coords.min(dim=1)[0]) * \
                   (y_coords.max(dim=1)[0] - y_coords.min(dim=1)[0])
            iou_kpt = kpt_iou(batch["keypoints"], preds["keypoints"], sigma=self.sigma, area=area)

            # For each prediction, find best matching GT
            for pi in range(len(preds["cls"])):
                if iou_kpt.shape[0] > 0:
                    iou_col = iou_kpt[:, pi] if iou_kpt.dim() == 2 else iou_kpt
                    best_iou = iou_col.max()

                    if best_iou > 0.5:
                        best_gt_idx = iou_col.argmax()
                        self.size_total += 1
                        gt_size_val = gt_size[best_gt_idx]
                        if gt_size_val.dim() > 0:
                            gt_size_val = gt_size_val.squeeze()
                        if pred_size[pi] == gt_size_val:
                            self.size_correct += 1

        return stats

    def finalize_metrics(self) -> None:
        """Finalize and log metrics including size accuracy."""
        super().finalize_metrics()

        # Log size accuracy
        if self.size_total > 0:
            size_acc = self.size_correct / self.size_total * 100
            LOGGER.info(f"Size accuracy: {size_acc:.2f}%")
