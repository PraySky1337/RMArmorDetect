"""
Armor Pose Loss - Loss function for armor detection with triple classification branches.

Uses WingLoss for keypoint regression (more sensitive to small errors than OKS).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

from ultralytics.utils.tal import make_anchors

from armor_detect.losses.wing_loss import WingLoss
from armor_detect.utils import TOP_K_ANCHORS, MAX_DIST_RATIO, FOCAL_GAMMA


class ArmorPoseLoss:
    """Loss function for armor pose detection with triple classification branches.

    This loss computes:
    - Keypoint regression loss using WingLoss (pose_loss)
    - Keypoint objectness loss (kobj_loss)
    - Number classification loss with Focal Loss (cls_loss)
    - Color classification loss with Focal Loss (color_loss)
    - Size classification loss with Focal Loss (size_loss)

    Label format: color size cls x1 y1 x2 y2 x3 y3 x4 y4
    where:
    - color: color class (0-3: B, R, N, P)
    - size: size class (0-1: s, b)
    - cls: number class (0-7: G, 1, 2, 3, 4, 5, O, B)
    - x1-x4, y1-y4: 4 corner keypoints (normalized coordinates)

    Args:
        model: The armor detection model (must have a Pose head with kpt_shape, nc_num, nc_color, nc_size)
    """

    def __init__(self, model):
        """Initialize ArmorPoseLoss with model configuration."""
        device = next(model.parameters()).device
        h = model.args  # hyperparameters

        # Find the Pose layer in the model
        pose_layer = None
        for layer in reversed(model.model):
            if hasattr(layer, "kpt_shape"):
                pose_layer = layer
                break
        if pose_layer is None:
            raise ValueError("Cannot find a layer with kpt_shape in model.model")

        self.kpt_shape = pose_layer.kpt_shape
        self.nc_num = pose_layer.nc_num
        self.nc_color = pose_layer.nc_color
        self.nc_size = getattr(pose_layer, 'nc_size', 2)
        self.nc = self.nc_num  # for compatibility

        self.device = device
        self.hyp = h
        self.pose_layer = pose_layer

        # WingLoss for keypoint regression (replaces OKS-based KeypointLoss)
        wing_omega = getattr(h, 'wing_omega', 10.0)
        wing_epsilon = getattr(h, 'wing_epsilon', 2.0)
        self.keypoint_loss = WingLoss(omega=wing_omega, epsilon=wing_epsilon)

        # Classification losses for all branches
        self.bce_num = nn.BCEWithLogitsLoss(reduction="none")
        self.bce_color = nn.BCEWithLogitsLoss(reduction="none")
        self.bce_size = nn.BCEWithLogitsLoss(reduction="none")

        # Focal Loss gamma for hard example mining
        self.focal_gamma = FOCAL_GAMMA

        # Positive sample assignment parameters
        self.topk = TOP_K_ANCHORS
        self.max_dist_ratio = MAX_DIST_RATIO

    def __call__(
        self, preds: tuple, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate losses for keypoint-only pose detection with triple classification branches.

        Args:
            preds: Model predictions
                - Training mode: (feats_list, (cls_num, cls_color, cls_size, kpt))
                - Validation mode: (combined_output, (feats_list, (cls_num, cls_color, cls_size, kpt)))
            batch: Batch data containing:
                - batch_idx: Batch indices for each GT
                - cls: Number class labels
                - keypoints: Ground truth keypoints (normalized)
                - color: Color class labels
                - size: Size class labels

        Returns:
            tuple: (total_loss * batch_size, loss_components)
                - loss_components: [pose_loss, kobj_loss, cls_loss, color_loss, size_loss]
        """
        loss = torch.zeros(5, device=self.device)

        # Unpack predictions
        if isinstance(preds[0], list):
            # Training mode
            feats = preds[0]
            cls_num_pred, cls_color_pred, cls_size_pred, kpt_pred = preds[1]
        else:
            # Validation mode
            feats = preds[1][0]
            cls_num_pred, cls_color_pred, cls_size_pred, kpt_pred = preds[1][1]

        # Permute dimensions for per-anchor loss computation
        cls_num_pred = cls_num_pred.permute(0, 2, 1)  # (bs, na, nc_num)
        cls_color_pred = cls_color_pred.permute(0, 2, 1)  # (bs, na, nc_color)
        cls_size_pred = cls_size_pred.permute(0, 2, 1)  # (bs, na, nc_size)
        kpt_pred = kpt_pred.permute(0, 2, 1)  # (bs, na, nk)

        # Get stride dynamically from pose_layer
        stride = self.pose_layer.stride
        if stride.sum() == 0:
            # Fallback: compute stride from image size and feature map size
            img_h, img_w = batch["img"].shape[2:]
            stride = torch.tensor(
                [img_h / f.shape[2] for f in feats], device=self.device
            )

        imgsz = torch.tensor(feats[0].shape[2:], device=self.device) * stride[0]
        img_h, img_w = batch["img"].shape[2:]
        self.imgsz = torch.tensor([img_h, img_w], device=self.device)
        anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)

        # Process targets
        batch_size = cls_num_pred.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        gt_cls_num = batch["cls"].view(-1, 1)
        num_anchors = anchor_points.shape[0]

        # Initialize assignment tensors
        fg_mask = torch.zeros(
            batch_size, num_anchors, device=self.device, dtype=torch.bool
        )
        target_gt_idx = torch.full(
            (batch_size, num_anchors), -1, device=self.device, dtype=torch.long
        )
        target_scores = torch.zeros(batch_size, num_anchors, 1, device=self.device)

        if "keypoints" in batch:
            keypoints = batch["keypoints"].to(self.device).float()

            # Convert normalized keypoints to pixel coordinates
            keypoints_px = keypoints.clone()
            keypoints_px[..., 0] *= imgsz[1]
            keypoints_px[..., 1] *= imgsz[0]

            # Convert anchor points to pixel coordinates
            anchor_points_px = anchor_points * stride_tensor

            batch_idx_flat = batch_idx.view(-1)
            gt_kpt_centers = keypoints_px.mean(dim=1)  # GT keypoint centers

            # Assign positive samples using top-k nearest anchors
            batch_gt_counters = {}
            for gt_idx, (b_idx, center) in enumerate(
                zip(batch_idx_flat.tolist(), gt_kpt_centers.tolist())
            ):
                b_idx = int(b_idx)

                if b_idx not in batch_gt_counters:
                    batch_gt_counters[b_idx] = 0
                per_batch_idx = batch_gt_counters[b_idx]
                batch_gt_counters[b_idx] += 1

                # Find top-k closest anchors
                center_tensor = torch.tensor(center, device=self.device)
                dists = torch.norm(anchor_points_px - center_tensor, dim=1)

                # Distance threshold based on stride
                max_dist = stride_tensor.squeeze() * self.max_dist_ratio
                valid_dist_mask = dists < max_dist.squeeze()

                valid_dists = dists.clone()
                valid_dists[~valid_dist_mask] = float("inf")

                k = min(self.topk, len(dists))
                _, topk_indices = dists.topk(k, largest=False)

                for anchor_idx in topk_indices:
                    anchor_idx = int(anchor_idx.item())
                    if not fg_mask[b_idx, anchor_idx]:
                        fg_mask[b_idx, anchor_idx] = True
                        target_gt_idx[b_idx, anchor_idx] = per_batch_idx
                        target_scores[b_idx, anchor_idx, 0] = 1.0

            # Compute keypoint loss
            if fg_mask.sum() > 0:
                kpt_pred_decoded = self.decode_keypoints(
                    kpt_pred, anchor_points, stride_tensor
                )
                loss[0], loss[1] = self.calculate_keypoints_loss(
                    fg_mask,
                    target_gt_idx,
                    keypoints_px,
                    batch_idx,
                    stride_tensor,
                    kpt_pred_decoded,
                )
            else:
                loss[0] = (kpt_pred * 0).sum()
                loss[1] = (kpt_pred * 0).sum()
        else:
            loss[0] = (kpt_pred * 0).sum()
            loss[1] = (kpt_pred * 0).sum()

        # Number classification loss with Focal Loss
        num_class_targets = torch.zeros_like(cls_num_pred)
        batch_idx_for_cls = batch_idx.view(-1)

        for b in range(batch_size):
            valid_mask = fg_mask[b]
            if valid_mask.sum() > 0:
                batch_mask = batch_idx_for_cls == b
                batch_cls = gt_cls_num[batch_mask]
                gt_indices = target_gt_idx[b][valid_mask]
                valid_anchor_indices = torch.where(valid_mask)[0]

                for i, gt_idx in enumerate(gt_indices):
                    gt_idx = int(gt_idx.item())
                    if 0 <= gt_idx < len(batch_cls):
                        class_id = int(batch_cls[gt_idx].item())
                        if class_id < self.nc_num:
                            anchor_idx = valid_anchor_indices[i]
                            num_class_targets[b, anchor_idx, class_id] = 1.0

        if fg_mask.sum() > 0:
            fg_cls_pred = cls_num_pred[fg_mask]
            fg_cls_target = num_class_targets[fg_mask]
            cls_loss = self.bce_num(fg_cls_pred, fg_cls_target.to(fg_cls_pred.dtype))
            # Focal Loss weighting
            pt = torch.exp(-cls_loss)
            focal_weight = (1 - pt) ** self.focal_gamma
            loss[2] = (cls_loss * focal_weight).mean()
        else:
            loss[2] = (cls_num_pred * 0).sum()

        # Color classification loss with Focal Loss
        if "color" in batch:
            gt_color = batch["color"].view(-1)
            color_targets = torch.zeros_like(cls_color_pred)

            for b in range(batch_size):
                valid_mask = fg_mask[b]
                if valid_mask.sum() > 0:
                    batch_mask = batch_idx_for_cls == b
                    batch_color = gt_color[batch_mask]
                    gt_indices = target_gt_idx[b][valid_mask]
                    valid_anchor_indices = torch.where(valid_mask)[0]

                    for i, gt_idx in enumerate(gt_indices):
                        gt_idx = int(gt_idx.item())
                        if 0 <= gt_idx < len(batch_color):
                            color_id = int(batch_color[gt_idx].item())
                            if color_id < self.nc_color:
                                anchor_idx = valid_anchor_indices[i]
                                color_targets[b, anchor_idx, color_id] = 1.0

            if fg_mask.sum() > 0:
                fg_color_pred = cls_color_pred[fg_mask]
                fg_color_target = color_targets[fg_mask]
                color_loss = self.bce_color(
                    fg_color_pred, fg_color_target.to(fg_color_pred.dtype)
                )
                pt = torch.exp(-color_loss)
                focal_weight = (1 - pt) ** self.focal_gamma
                loss[3] = (color_loss * focal_weight).mean()
            else:
                loss[3] = (cls_color_pred * 0).sum()
        else:
            loss[3] = (cls_color_pred * 0).sum()

        # Size classification loss with Focal Loss
        if "size" in batch:
            gt_size = batch["size"].view(-1)
            size_targets = torch.zeros_like(cls_size_pred)

            for b in range(batch_size):
                valid_mask = fg_mask[b]
                if valid_mask.sum() > 0:
                    batch_mask = batch_idx_for_cls == b
                    batch_size_labels = gt_size[batch_mask]
                    gt_indices = target_gt_idx[b][valid_mask]
                    valid_anchor_indices = torch.where(valid_mask)[0]

                    for i, gt_idx in enumerate(gt_indices):
                        gt_idx = int(gt_idx.item())
                        if 0 <= gt_idx < len(batch_size_labels):
                            size_id = int(batch_size_labels[gt_idx].item())
                            if size_id < self.nc_size:
                                anchor_idx = valid_anchor_indices[i]
                                size_targets[b, anchor_idx, size_id] = 1.0

            if fg_mask.sum() > 0:
                fg_size_pred = cls_size_pred[fg_mask]
                fg_size_target = size_targets[fg_mask]
                size_loss = self.bce_size(
                    fg_size_pred, fg_size_target.to(fg_size_pred.dtype)
                )
                pt = torch.exp(-size_loss)
                focal_weight = (1 - pt) ** self.focal_gamma
                loss[4] = (size_loss * focal_weight).mean()
            else:
                loss[4] = (cls_size_pred * 0).sum()
        else:
            loss[4] = (cls_size_pred * 0).sum()

        # Apply loss weights from hyperparameters
        loss[0] *= self.hyp.pose
        loss[1] *= self.hyp.kobj
        loss[2] *= self.hyp.cls
        loss[3] *= getattr(self.hyp, "color", 1.0)
        loss[4] *= getattr(self.hyp, "size", 1.0)

        return loss * batch_size, loss.detach()

    def decode_keypoints(
        self,
        kpt_pred: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Decode raw keypoint predictions to pixel coordinates.

        Args:
            kpt_pred: Raw network output (bs, na, nk)
            anchor_points: Anchor coordinates (na, 2)
            stride_tensor: Stride for each anchor (na, 1)

        Returns:
            Decoded keypoints in pixel coordinates (bs, na, nk)
        """
        bs, na, nk = kpt_pred.shape
        nkpt = self.kpt_shape[0]
        ndim = self.kpt_shape[1]

        kpt = kpt_pred.view(bs, na, nkpt, ndim)

        anchor_x = anchor_points[:, 0].view(1, -1, 1)
        anchor_y = anchor_points[:, 1].view(1, -1, 1)
        strides = stride_tensor.view(1, -1, 1)

        # Decode: (pred * 2.0 + (anchor - 0.5)) * stride
        kpt_x = (kpt[..., 0] * 2.0 + (anchor_x - 0.5)) * strides
        kpt_y = (kpt[..., 1] * 2.0 + (anchor_y - 0.5)) * strides

        # Clamp to image bounds
        if hasattr(self, "imgsz") and self.imgsz is not None:
            kpt_x = kpt_x.clamp(0, self.imgsz[1])
            kpt_y = kpt_y.clamp(0, self.imgsz[0])

        kpt = torch.stack([kpt_x, kpt_y], dim=-1)
        return kpt.view(bs, na, nk)

    def calculate_keypoints_loss(
        self,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate keypoint loss using WingLoss.

        Args:
            fg_mask: Foreground mask (bs, na)
            target_gt_idx: Ground truth indices (bs, na)
            keypoints: GT keypoints in pixel coords (total_objects, nkpt, ndim)
            batch_idx: Batch indices (total_objects,)
            stride_tensor: Stride tensor (na, 1)
            pred_kpts: Predicted keypoints (bs, na, nk)

        Returns:
            tuple: (pose_loss, kobj_loss)
        """
        loss_pose = torch.zeros(1, device=self.device)
        loss_kobj = torch.zeros(1, device=self.device)

        bs, na, nk = pred_kpts.shape
        nkpt = self.kpt_shape[0]
        ndim = self.kpt_shape[1]

        # Get predictions for foreground anchors
        pred_kpts_fg = pred_kpts[fg_mask].view(-1, nkpt, ndim)

        # Build target keypoints for each foreground anchor
        target_kpts_list = []
        areas_list = []
        batch_idx_flat = batch_idx.view(-1)

        for b in range(bs):
            batch_mask = batch_idx_flat == b
            batch_kpts = keypoints[batch_mask]

            fg_in_batch = fg_mask[b]
            gt_indices = target_gt_idx[b][fg_in_batch]

            for gt_idx in gt_indices:
                gt_idx = int(gt_idx.item())
                if 0 <= gt_idx < len(batch_kpts):
                    kpts = batch_kpts[gt_idx]
                    target_kpts_list.append(kpts)

                    # Calculate area from keypoint bounding box
                    kpt_xs = kpts[:, 0]
                    kpt_ys = kpts[:, 1]
                    w = (kpt_xs.max() - kpt_xs.min()).clamp(min=1.0)
                    h = (kpt_ys.max() - kpt_ys.min()).clamp(min=1.0)
                    areas_list.append(w * h)

        if len(target_kpts_list) == 0:
            return loss_pose, loss_kobj

        target_kpts = torch.stack(target_kpts_list)
        areas = torch.stack(areas_list)

        if pred_kpts_fg.shape[0] != target_kpts.shape[0]:
            return loss_pose, loss_kobj

        # Flatten keypoints for WingLoss: (N, nkpt, 2) -> (N, nkpt*2)
        pred_flat = pred_kpts_fg.reshape(-1, nkpt * ndim)
        target_flat = target_kpts.reshape(-1, nkpt * ndim)

        # Normalize by stride/scale for better loss magnitude
        # WingLoss works with pixel coordinates directly
        loss_pose = self.keypoint_loss(pred_flat, target_flat)

        if not torch.isfinite(loss_pose):
            loss_pose = (pred_kpts_fg * 0).sum()

        # Keypoint objectness loss based on distance
        with torch.no_grad():
            d = (pred_kpts_fg[..., :2] - target_kpts[..., :2]).pow(2).sum(dim=-1)
            s = areas.unsqueeze(-1)
            # OKS-like metric for objectness target
            sigmas = torch.ones(nkpt, device=self.device) / nkpt
            kpt_oks = torch.exp(-d / (2 * s * (sigmas**2) + 1e-9))
            oks = kpt_oks.mean(dim=-1)

        loss_kobj = ((1.0 - oks) ** 2).mean()

        return loss_pose.unsqueeze(0), loss_kobj.unsqueeze(0)
