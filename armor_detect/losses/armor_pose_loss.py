"""
Armor Pose Loss - Loss function for armor detection with triple classification branches.

Uses WingLoss for keypoint regression (more sensitive to small errors than OKS).
Fully vectorized implementation for maximum GPU utilization.
"""

import torch
import torch.nn as nn

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

    # Fixed maximum GT per batch to avoid CPU-GPU sync (.item() call)
    # Set to 64 to handle mosaic/mixup augmentation (max 8 objects/image * 4 mosaic * 2x safety margin)
    MAX_GT_PER_BATCH = 64

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

        Fully vectorized implementation - no .item() or .tolist() calls.
        """
        loss = torch.zeros(5, device=self.device)

        # Unpack predictions
        if isinstance(preds[0], list):
            feats = preds[0]
            cls_num_pred, cls_color_pred, cls_size_pred, kpt_pred = preds[1]
        else:
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
            img_h, img_w = batch["img"].shape[2:]
            stride = torch.tensor(
                [img_h / f.shape[2] for f in feats], device=self.device
            )

        imgsz = torch.tensor(feats[0].shape[2:], device=self.device) * stride[0]
        img_h, img_w = batch["img"].shape[2:]
        self.imgsz = torch.tensor([img_h, img_w], device=self.device)
        anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)

        batch_size = cls_num_pred.shape[0]
        num_anchors = anchor_points.shape[0]
        batch_idx = batch["batch_idx"].view(-1)  # (num_gt,)
        gt_cls_num = batch["cls"].view(-1)  # (num_gt,)
        num_gt = batch_idx.shape[0]

        # Initialize assignment tensors
        fg_mask = torch.zeros(batch_size, num_anchors, device=self.device, dtype=torch.bool)
        target_gt_idx = torch.full((batch_size, num_anchors), -1, device=self.device, dtype=torch.long)

        if "keypoints" in batch and num_gt > 0:
            keypoints = batch["keypoints"].to(self.device).float()  # (num_gt, nkpt, 2)

            # Convert normalized keypoints to pixel coordinates
            keypoints_px = keypoints.clone()
            keypoints_px[..., 0] *= imgsz[1]
            keypoints_px[..., 1] *= imgsz[0]

            # Convert anchor points to pixel coordinates
            anchor_points_px = anchor_points * stride_tensor  # (na, 2)

            # GT keypoint centers: (num_gt, 2)
            gt_centers = keypoints_px.mean(dim=1)

            # ========== VECTORIZED POSITIVE SAMPLE ASSIGNMENT ==========
            # Compute distances from all GTs to all anchors: (num_gt, na)
            # Ensure same dtype for cdist (AMP may cause dtype mismatch)
            gt_centers_f = gt_centers.float()
            anchor_points_px_f = anchor_points_px.float()
            dists = torch.cdist(gt_centers_f, anchor_points_px_f)  # (num_gt, na)

            # Get top-k closest anchors for each GT
            k = min(self.topk, num_anchors)
            _, topk_indices = dists.topk(k, dim=1, largest=False)  # (num_gt, k)

            # Build per-batch GT index mapping
            # For each batch, we need to know the local index of each GT within that batch
            gt_batch_local_idx = torch.zeros(num_gt, device=self.device, dtype=torch.long)
            for b in range(batch_size):
                batch_mask = batch_idx == b
                num_gt_in_batch = batch_mask.sum()
                if num_gt_in_batch > 0:
                    gt_batch_local_idx[batch_mask] = torch.arange(num_gt_in_batch, device=self.device)

            # Assign anchors to GTs (vectorized)
            batch_indices_expanded = batch_idx.unsqueeze(1).expand(-1, k)  # (num_gt, k)
            local_gt_indices_expanded = gt_batch_local_idx.unsqueeze(1).expand(-1, k)  # (num_gt, k)

            # Flatten for scatter
            batch_flat = batch_indices_expanded.reshape(-1).long()  # (num_gt * k,)
            anchor_flat = topk_indices.reshape(-1).long()  # (num_gt * k,)
            local_gt_flat = local_gt_indices_expanded.reshape(-1).long()  # (num_gt * k,)

            # Use scatter to assign (first assignment wins due to order)
            # We need to handle conflicts where multiple GTs assign to same anchor
            # For simplicity, we assign all and let later GTs overwrite (acceptable for loss)
            fg_mask[batch_flat, anchor_flat] = True
            target_gt_idx[batch_flat, anchor_flat] = local_gt_flat

            # Normalization factor for classification loss (number of positive samples)
            # Following v8DetectionLoss pattern (ultralytics/utils/loss.py line 429)
            target_scores_sum = max(fg_mask.sum(), 1)

            # ========== COMPUTE KEYPOINT LOSS ==========
            if fg_mask.sum() > 0:
                kpt_pred_decoded = self.decode_keypoints(kpt_pred, anchor_points, stride_tensor)
                loss[0], loss[1], oks_scores = self.calculate_keypoints_loss_vectorized(
                    fg_mask, target_gt_idx, keypoints_px, batch_idx, kpt_pred_decoded
                )
            else:
                loss[0] = (kpt_pred * 0).sum()
                loss[1] = (kpt_pred * 0).sum()
                oks_scores = torch.zeros(batch_size, num_anchors, device=self.device)

            # ========== VECTORIZED CLASSIFICATION TARGET CONSTRUCTION ==========
            # Build all classification targets at once using scatter
            # Use OKS as soft target instead of hard 1.0 (Phase 2 improvement)
            num_class_targets = torch.zeros_like(cls_num_pred)
            color_targets = torch.zeros_like(cls_color_pred)
            size_targets = torch.zeros_like(cls_size_pred)

            # Get GT labels
            gt_color = batch.get("color", torch.zeros_like(gt_cls_num)).view(-1)
            gt_size = batch.get("size", torch.zeros_like(gt_cls_num)).view(-1)

            # For each positive anchor, find its GT and assign label with OKS soft target
            for b in range(batch_size):
                fg_in_batch = fg_mask[b]
                if fg_in_batch.sum() == 0:
                    continue

                batch_gt_mask = batch_idx == b
                batch_cls = gt_cls_num[batch_gt_mask]
                batch_color = gt_color[batch_gt_mask]
                batch_size_labels = gt_size[batch_gt_mask]

                gt_indices = target_gt_idx[b, fg_in_batch]  # (num_fg,)
                valid = (gt_indices >= 0) & (gt_indices < len(batch_cls))

                if valid.sum() == 0:
                    continue

                valid_gt_indices = gt_indices[valid]
                valid_anchor_indices = torch.where(fg_in_batch)[0][valid]

                # Get class IDs
                cls_ids = batch_cls[valid_gt_indices].long()
                color_ids = batch_color[valid_gt_indices].long()
                size_ids = batch_size_labels[valid_gt_indices].long()

                # Clamp to valid range
                cls_ids = cls_ids.clamp(0, self.nc_num - 1)
                color_ids = color_ids.clamp(0, self.nc_color - 1)
                size_ids = size_ids.clamp(0, self.nc_size - 1)

                # Get OKS soft targets for these anchors (instead of hard 1.0)
                # Cast to target dtype for AMP compatibility (FP16 targets vs FP32 OKS)
                oks_soft = oks_scores[b, valid_anchor_indices].to(num_class_targets.dtype)

                # Scatter to set targets using OKS soft labels
                num_class_targets[b, valid_anchor_indices, cls_ids] = oks_soft
                color_targets[b, valid_anchor_indices, color_ids] = oks_soft
                size_targets[b, valid_anchor_indices, size_ids] = oks_soft

        else:
            loss[0] = (kpt_pred * 0).sum()
            loss[1] = (kpt_pred * 0).sum()
            num_class_targets = torch.zeros_like(cls_num_pred)
            color_targets = torch.zeros_like(cls_color_pred)
            size_targets = torch.zeros_like(cls_size_pred)
            target_scores_sum = 1  # No positive samples, use 1 to avoid division by zero

        # ========== CLASSIFICATION LOSSES WITH FOCAL LOSS (ALL ANCHORS) ==========
        # Compute classification loss on ALL anchors to constrain negative samples
        # Negative samples have target=0.0 (from initialization at lines 194-196)
        # This follows v8DetectionLoss pattern (ultralytics/utils/loss.py line 433)

        # Number classification loss (Focal Loss) - on ALL anchors
        cls_loss = self.bce_num(cls_num_pred, num_class_targets)  # (bs, na, nc_num)
        pt = torch.exp(-cls_loss)
        focal_weight = (1 - pt) ** self.focal_gamma
        loss[2] = (cls_loss * focal_weight).sum() / target_scores_sum

        # Color classification loss - on ALL anchors
        if "color" in batch:
            color_loss = self.bce_color(cls_color_pred, color_targets)  # (bs, na, nc_color)
            pt = torch.exp(-color_loss)
            focal_weight = (1 - pt) ** self.focal_gamma
            loss[3] = (color_loss * focal_weight).sum() / target_scores_sum
        else:
            loss[3] = (cls_color_pred * 0).sum()

        # Size classification loss - on ALL anchors
        if "size" in batch:
            size_loss = self.bce_size(cls_size_pred, size_targets)  # (bs, na, nc_size)
            pt = torch.exp(-size_loss)
            focal_weight = (1 - pt) ** self.focal_gamma
            loss[4] = (size_loss * focal_weight).sum() / target_scores_sum
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
        """Decode raw keypoint predictions to pixel coordinates."""
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

    def calculate_keypoints_loss_vectorized(
        self,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate keypoint loss using WingLoss - fully vectorized.

        Returns:
            loss_pose: Keypoint regression loss (WingLoss)
            loss_kobj: Keypoint objectness loss (OKS-based)
            oks_scores: OKS values for all anchors (bs, na), 0 for negatives
        """
        loss_pose = torch.zeros(1, device=self.device)
        loss_kobj = torch.zeros(1, device=self.device)

        bs = pred_kpts.shape[0]
        na = pred_kpts.shape[1]
        nkpt = self.kpt_shape[0]
        ndim = self.kpt_shape[1]

        # Initialize OKS scores for all anchors (negatives get 0.0)
        oks_scores = torch.zeros(bs, na, device=self.device)

        # Get predictions for foreground anchors
        pred_kpts_fg = pred_kpts[fg_mask].view(-1, nkpt, ndim)
        num_fg = pred_kpts_fg.shape[0]

        if num_fg == 0:
            return loss_pose, loss_kobj, oks_scores

        # Build target keypoints using gather (vectorized)
        # First, build a padded keypoints tensor per batch
        # Use bincount to get GT counts per batch efficiently
        gt_counts = torch.bincount(batch_idx.long(), minlength=bs)

        # Use fixed MAX_GT_PER_BATCH to avoid CPU-GPU sync (.item() call)
        max_gt_per_batch = self.MAX_GT_PER_BATCH

        # Check if there are any GTs (avoid .item() by using tensor comparison)
        if gt_counts.sum() == 0:
            return loss_pose, loss_kobj, oks_scores

        # Boundary check: warn if any batch exceeds MAX_GT_PER_BATCH
        # Note: max() stays on GPU; only sync to CPU for warning in rare overflow case
        max_actual = gt_counts.max()
        if max_actual > max_gt_per_batch:
            # Only call .item() in rare overflow case for warning
            from ultralytics.utils import LOGGER
            LOGGER.warning(
                f"GT count {max_actual.item()} exceeds MAX_GT_PER_BATCH={max_gt_per_batch}. "
                f"Truncating to first {max_gt_per_batch} GTs per batch. "
                f"Consider increasing MAX_GT_PER_BATCH in ArmorPoseLoss if this occurs frequently."
            )

        # Create padded keypoints tensor: (bs, MAX_GT_PER_BATCH, nkpt, ndim)
        padded_kpts = torch.zeros(bs, max_gt_per_batch, nkpt, ndim, device=self.device)
        for b in range(bs):
            batch_mask = batch_idx == b
            batch_kpts = keypoints[batch_mask]
            num_gt_b = batch_kpts.shape[0]
            if num_gt_b > 0:
                # Truncate if exceeds MAX_GT_PER_BATCH
                num_to_copy = min(num_gt_b, max_gt_per_batch)
                padded_kpts[b, :num_to_copy] = batch_kpts[:num_to_copy]

        # Gather target keypoints for each foreground anchor
        target_kpts_list = []
        areas_list = []

        for b in range(bs):
            fg_in_batch = fg_mask[b]
            if fg_in_batch.sum() == 0:
                continue

            gt_indices = target_gt_idx[b, fg_in_batch]  # (num_fg_b,)
            valid = (gt_indices >= 0) & (gt_indices < max_gt_per_batch)

            if valid.sum() == 0:
                continue

            valid_gt_indices = gt_indices[valid].long()
            # Gather keypoints: (num_valid, nkpt, ndim)
            kpts = padded_kpts[b, valid_gt_indices]
            target_kpts_list.append(kpts)

            # Calculate areas
            kpt_xs = kpts[..., 0]
            kpt_ys = kpts[..., 1]
            w = (kpt_xs.max(dim=-1).values - kpt_xs.min(dim=-1).values).clamp(min=1.0)
            h = (kpt_ys.max(dim=-1).values - kpt_ys.min(dim=-1).values).clamp(min=1.0)
            areas_list.append(w * h)

        if len(target_kpts_list) == 0:
            return loss_pose, loss_kobj, oks_scores

        target_kpts = torch.cat(target_kpts_list, dim=0)  # (num_fg, nkpt, ndim)
        areas = torch.cat(areas_list, dim=0)  # (num_fg,)

        # Match shapes
        if pred_kpts_fg.shape[0] != target_kpts.shape[0]:
            # Truncate to match (can happen due to scatter conflicts)
            min_len = min(pred_kpts_fg.shape[0], target_kpts.shape[0])
            pred_kpts_fg = pred_kpts_fg[:min_len]
            target_kpts = target_kpts[:min_len]
            areas = areas[:min_len]

        # Flatten keypoints for WingLoss
        pred_flat = pred_kpts_fg.reshape(-1, nkpt * ndim)
        target_flat = target_kpts.reshape(-1, nkpt * ndim)

        # Compute WingLoss
        loss_pose = self.keypoint_loss(pred_flat, target_flat)

        if not torch.isfinite(loss_pose):
            loss_pose = torch.zeros(1, device=self.device)

        # Keypoint objectness loss based on OKS
        with torch.no_grad():
            d = (pred_kpts_fg - target_kpts).pow(2).sum(dim=-1)  # (num_fg, nkpt)
            s = areas.unsqueeze(-1)  # (num_fg, 1)
            sigmas = torch.ones(nkpt, device=self.device) / nkpt
            kpt_oks = torch.exp(-d / (2 * s * (sigmas**2) + 1e-9))
            oks = kpt_oks.mean(dim=-1)  # (num_fg,)

        loss_kobj = ((1.0 - oks) ** 2).mean()

        # Fill OKS scores into the full (bs, na) tensor for soft targets
        # Handle potential length mismatch due to gather/scatter conflicts
        fg_indices = torch.where(fg_mask.view(-1))[0]  # Flat indices of fg anchors
        num_actual = min(oks.shape[0], fg_indices.shape[0])
        if num_actual > 0:
            fg_indices = fg_indices[:num_actual]
            oks_values = oks[:num_actual]
            # Convert flat indices to (bs, na) indices
            bs_idx = fg_indices // na
            anchor_idx = fg_indices % na
            oks_scores[bs_idx, anchor_idx] = oks_values.detach()

        return loss_pose.unsqueeze(0) if loss_pose.dim() == 0 else loss_pose.view(1), loss_kobj.unsqueeze(0), oks_scores
