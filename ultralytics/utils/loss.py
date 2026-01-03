# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing on
    hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int = 16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max: int):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        #欧氏距离平方
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        #关键点有效性归一化因子（对有效关键点少的样本放大补偿）
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        #关键点误差归一化公式 
        s = torch.sqrt(area).unsqueeze(-1)  # (N,1)
        e = d / (2 * (s * self.sigmas) ** 2 + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        # e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()

# class KeypointLoss(nn.Module):
#     """Criterion class for computing keypoint losses (pixel-coordinate version)."""

#     def __init__(self, sigmas: torch.Tensor) -> None:
#         """Initialize the KeypointLoss class with keypoint sigmas (in pixel units)."""
#         super().__init__()
#         self.sigmas = sigmas  # shape: (num_keypoints,)

#     def forward(
#         self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor = None
#     ) -> torch.Tensor:
#         """
#         Calculate keypoint loss factor and Euclidean distance loss for keypoints.

#         Args:
#             pred_kpts: (batch, num_kpts, 2) tensor, predicted keypoints in pixel coordinates
#             gt_kpts: (batch, num_kpts, 2) tensor, ground truth keypoints in pixel coordinates
#             kpt_mask: (batch, num_kpts) tensor, 1 if keypoint is visible, 0 otherwise
#             area: optional (batch,) tensor, area for each person in pixels (if None, compute from keypoints)
#         """
#         batch_size, num_kpts, _ = pred_kpts.shape

#         # --- 调试打印 ---
#         # print("pred_kpts[0]:", pred_kpts[0])
#         # print("gt_kpts[0]:", gt_kpts[0])
#         # print("kpt_mask[0]:", kpt_mask[0])

#         # --- 计算目标面积（如果没有传入） ---
#         if area is None:
#             x_min, _ = gt_kpts[..., 0].min(dim=1)
#             x_max, _ = gt_kpts[..., 0].max(dim=1)
#             y_min, _ = gt_kpts[..., 1].min(dim=1)
#             y_max, _ = gt_kpts[..., 1].max(dim=1)
#             area = (x_max - x_min) * (y_max - y_min)
#         print("area[0]:", area[0])

#         # --- 欧氏距离平方 ---
#         d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)

#         # --- 有效关键点归一化因子 ---
#         kpt_loss_factor = num_kpts / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)

#         # --- 广播 sigmas 和 area ---
#         sigmas = self.sigmas.view(1, num_kpts)  # (1, num_kpts)
#         area = area.view(batch_size, 1)        # (batch, 1)

#         # --- 关键点误差归一化（pixel coordinate version） ---
#         e = d / (2 * (area * sigmas) ** 2 + 1e-9)

#         # --- loss ---
#         loss = kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)
#         return loss.mean()



class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, model, tal_topk: int = 10):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        loss = torch.zeros(4, device=self.device)  # box, seg, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, seg, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (N,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()
    
class v8PoseLoss(v8DetectionLoss):
    """Criterion class for YOLOv8 Pose with dual classification branches (number and color).

    Keypoint-only mode (no bbox regression).

    Label format: color cls x1 y1 x2 y2 x3 y3 x4 y4
    where:
    - color: color class (0-3: R, B, N, P)
    - cls: number class (0-7: numbers 1-8)
    - x1-x4, y1-y4: 4 corner keypoints
    """

    def __init__(self, model):
        """Initialize v8PoseLoss with keypoint and dual classification branches."""
        device = next(model.parameters()).device 
        h = model.args  # hyperparameters超参数集合

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
        self.nc = self.nc_num  

        self.device = device
        self.hyp = h
        self.pose_layer = pose_layer  # Keep reference to get stride dynamically
        

        # Keypoint loss和分类损失初始化
        nkpt = self.kpt_shape[0]
        sigmas = torch.ones(nkpt, device=device) / nkpt #每个关键点的标准差，影响OKS权重
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

        # Classification losses for both branches
        self.bce_num = nn.BCEWithLogitsLoss(reduction="none")  # number branch
        self.bce_color = nn.BCEWithLogitsLoss(reduction="none")  # color branch

        # Focal Loss gamma for hard example mining
        self.focal_gamma = 1.5  # 温和的gamma值，平衡精度和召回

    #前向计算
    def __call__(self, preds, batch):
        """
        Calculate losses for keypoint-only pose detection with dual classification branches.

        preds: (feats, (cls_num, cls_color, kpt)) during inference
        batch: contains batch_idx, cls (number), keypoints, color
        loss: [pose, kobj, cls_num, cls_color] (no bbox/dfl losses)
        """
        loss = torch.zeros(4, device=self.device) #初始化损失[pose, kobj, cls_num, cls_color]

        # 预测解包
        # Training mode: preds = (feats_list, (cls_num, cls_color, kpt))
        # Validation mode: preds = (combined_output, (feats_list, (cls_num, cls_color, kpt)))
        if isinstance(preds[0], list):
            # Training mode
            feats = preds[0] #特征图列表
            cls_num_pred, cls_color_pred, kpt_pred = preds[1]
        else:
            # Validation mode - preds[1][0] should be feats list
            feats = preds[1][0]
            cls_num_pred, cls_color_pred, kpt_pred = preds[1][1]

        #permute调整维度方便按anchor计算损失
        cls_num_pred = cls_num_pred.permute(0, 2, 1)  # (bs, na, nc_num)
        cls_color_pred = cls_color_pred.permute(0, 2, 1)  # (bs, na, nc_color)
        kpt_pred = kpt_pred.permute(0, 2, 1)  # (bs, na, nk)

        # Get stride dynamically from pose_layer (it's computed during first forward pass)
        #动态计算stride，stride用于将anchor映射回像素坐标，如果低碳日的尚未初始化就按img_size / feat_map_size 动态计算
        stride = self.pose_layer.stride
        if stride.sum() == 0:
            # Fallback: compute stride from image size and feature map size
            img_h, img_w = batch["img"].shape[2:]  # (bs, c, h, w)
            stride = torch.tensor([img_h / f.shape[2] for f in feats], device=self.device)
            # print(f"[DEBUG] Computed stride from img/feats: {stride.tolist()}")


        imgsz = torch.tensor(feats[0].shape[2:], device=self.device) * stride[0]
        # Use actual image size for clamp bounds (more accurate than feats * stride)
        img_h, img_w = batch["img"].shape[2:]
        self.imgsz = torch.tensor([img_h, img_w], device=self.device)
        anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)

        # ---------- process targets ----------
        batch_size = cls_num_pred.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        gt_cls_num = batch["cls"].view(-1, 1)

        num_anchors = anchor_points.shape[0]

        fg_mask = torch.zeros(batch_size, num_anchors, device=self.device, dtype=torch.bool)
        target_gt_idx = torch.full((batch_size, num_anchors), -1, device=self.device, dtype=torch.long)
        target_scores = torch.zeros(batch_size, num_anchors, 1, device=self.device)

        if "keypoints" in batch:
            keypoints = batch["keypoints"].to(self.device).float()

            # if keypoints.numel() == 0:
            #     # No keypoints in this batch - return zero losses
            #     return loss.sum() * 0, torch.zeros(4, device=self.device)

            keypoints_px = keypoints.clone()
            keypoints_px[..., 0] *= imgsz[1]
            keypoints_px[..., 1] *= imgsz[0]

            # anchor_points is in feature map coordinates, need to multiply by stride
            anchor_points_px = anchor_points * stride_tensor  # (na, 2) * (na, 1) -> (na, 2)

            batch_idx_flat = batch_idx.view(-1)
            gt_kpt_centers = keypoints_px.mean(dim=1)  # (num_gt, 2) GT关键点中心

            batch_gt_counters = {}  # batch_id -> current count
            topk = 5  # 选top-k最近anchor作为正样本（从10减至5，温和收紧）
            max_dist_ratio = 1.2  # 距离阈值比例（从1.5减至1.2，温和收紧）

            #计算每个GT到所有anchor的真实距离
            for gt_idx, (b_idx, center) in enumerate(zip(batch_idx_flat.tolist(), gt_kpt_centers.tolist())):
                b_idx = int(b_idx)

                # Get per-batch index
                if b_idx not in batch_gt_counters:
                    batch_gt_counters[b_idx] = 0
                per_batch_idx = batch_gt_counters[b_idx]
                batch_gt_counters[b_idx] += 1

                # Find top-k closest anchors to this keypoint center (both in pixel coords)
                center_tensor = torch.tensor(center, device=self.device)
                dists = torch.norm(anchor_points_px - center_tensor, dim=1)

                # 新增：计算距离阈值（基于 stride）
                max_dist = stride_tensor.squeeze() * max_dist_ratio  # 每个 anchor 的最大允许距离
                valid_dist_mask = dists < max_dist.squeeze()

                # 在距离阈值内选 top-k
                valid_dists = dists.clone()
                valid_dists[~valid_dist_mask] = float('inf')  # 超出距离的设为无穷大

                # Get top-k closest anchors
                k = min(topk, len(dists))
                _, topk_indices = dists.topk(k, largest=False)

                for anchor_idx in topk_indices:
                    anchor_idx = int(anchor_idx.item())
                    # Only assign if not already assigned or if this GT is closer
                    if not fg_mask[b_idx, anchor_idx]:
                        fg_mask[b_idx, anchor_idx] = True
                        target_gt_idx[b_idx, anchor_idx] = per_batch_idx
                        target_scores[b_idx, anchor_idx, 0] = 1.0

            # ---------- keypoint loss ----------
            num_fg = fg_mask.sum().item()
        
            if fg_mask.sum() > 0:
                # Decode predicted keypoints to pixel coordinates for loss computation
                # kpt_pred is raw network output, need to decode like in head.py
                kpt_pred_decoded = self.decode_keypoints(kpt_pred, anchor_points, stride_tensor) #将网络输出映射回像素坐标

                loss[0], loss[1] = self.calculate_keypoints_loss(
                    fg_mask, target_gt_idx, keypoints_px,
                    batch_idx, stride_tensor, kpt_pred_decoded
                )
            else:
                # No foreground anchors - use prediction tensor to maintain gradient graph
                loss[0] = (kpt_pred * 0).sum()
                loss[1] = (kpt_pred * 0).sum()

                # Debug: print loss values
                # if loss[0].item() == 0:
                #     print(f"[DEBUG] pose_loss=0 but fg_mask.sum()={num_fg}")
        else:
            # No keypoints in batch - use prediction tensor to maintain gradient graph
            loss[0] = (kpt_pred * 0).sum()
            loss[1] = (kpt_pred * 0).sum()

        # ---------- number class loss ----------
        ts = max(target_scores.sum(), 1)

        # Create one-hot encoded class targets for foreground anchors
        num_class_targets = torch.zeros_like(cls_num_pred)
        # Use batch_idx defined earlier (same length as gt_cls_num)
        batch_idx_for_cls = batch_idx.view(-1)

        for b in range(batch_size):
            valid_mask = fg_mask[b]
            if valid_mask.sum() > 0:
                batch_mask = (batch_idx_for_cls == b)
                batch_cls = gt_cls_num[batch_mask]
                gt_indices = target_gt_idx[b][valid_mask]

                # 获取 valid_mask 为 True 的 anchor 索引
                valid_anchor_indices = torch.where(valid_mask)[0]

                for i, gt_idx in enumerate(gt_indices):
                    gt_idx = int(gt_idx.item())
                    if gt_idx >= 0 and gt_idx < len(batch_cls):
                        class_id = int(batch_cls[gt_idx].item())
                        if class_id < self.nc_num:
                            anchor_idx = valid_anchor_indices[i]
                            num_class_targets[b, anchor_idx, class_id] = 1.0


        #Only compute cls loss for foreground anchors with Focal Loss
        if fg_mask.sum() > 0:
            fg_cls_pred = cls_num_pred[fg_mask]  # (num_fg, nc_num)
            fg_cls_target = num_class_targets[fg_mask]  # (num_fg, nc_num)
            cls_loss = self.bce_num(fg_cls_pred, fg_cls_target.to(fg_cls_pred.dtype))
            # Focal Loss: 降低易分样本权重，聚焦困难样本
            pt = torch.exp(-cls_loss)
            focal_weight = (1 - pt) ** self.focal_gamma
            loss[2] = (cls_loss * focal_weight).mean()
        else:
            # Use prediction tensor to maintain gradient graph
            loss[2] = (cls_num_pred * 0).sum()
        
    

        # ---------- color class loss ----------
        if "color" in batch:
            gt_color = batch["color"].view(-1)  # color class (0-3), flatten to 1D
            color_targets = torch.zeros_like(cls_color_pred)

           
            for b in range(batch_size):
                valid_mask = fg_mask[b]
                if valid_mask.sum() > 0:
                    batch_mask = (batch_idx_for_cls == b)
                    batch_color = gt_color[batch_mask]
                    gt_indices = target_gt_idx[b][valid_mask]

                    # 获取 valid_mask 为 True 的 anchor 索引
                    valid_anchor_indices = torch.where(valid_mask)[0]

                    for i, gt_idx in enumerate(gt_indices):
                        gt_idx = int(gt_idx.item())
                        if gt_idx >= 0 and gt_idx < len(batch_color):
                            color_id = int(batch_color[gt_idx].item())
                            if color_id < self.nc_color:
                                anchor_idx = valid_anchor_indices[i]
                                color_targets[b, anchor_idx, color_id] = 1.0


            # Only compute color loss for foreground anchors with Focal Loss
            if fg_mask.sum() > 0:
                fg_color_pred = cls_color_pred[fg_mask]
                fg_color_target = color_targets[fg_mask]
                color_loss = self.bce_color(fg_color_pred, fg_color_target.to(fg_color_pred.dtype))
                # Focal Loss: 降低易分样本权重，聚焦困难样本
                pt = torch.exp(-color_loss)
                focal_weight = (1 - pt) ** self.focal_gamma
                loss[3] = (color_loss * focal_weight).mean()
            else:
                # Use prediction tensor to maintain gradient graph
                loss[3] = (cls_color_pred * 0).sum()
        else:
            # Use prediction tensor to maintain gradient graph
            loss[3] = (cls_color_pred * 0).sum()

        loss[0] *= self.hyp.pose     
        loss[1] *= self.hyp.kobj      
        loss[2] *= self.hyp.cls      
        loss[3] *= getattr(self.hyp, "color", 1.0) 

        return loss * batch_size, loss.detach()

    def decode_keypoints(self, kpt_pred, anchor_points, stride_tensor):
        """Decode raw keypoint predictions to pixel coordinates.

        Args:
            kpt_pred: (bs, na, nk) - raw network output
            anchor_points: (na, 2) - anchor coordinates
            stride_tensor: (na, 1) - stride for each anchor

        Returns:
            decoded: (bs, na, nk) - keypoints in pixel coordinates
        """
        bs, na, nk = kpt_pred.shape
        nkpt = self.kpt_shape[0]
        ndim = self.kpt_shape[1]

        # Reshape to (bs, na, nkpt, ndim)
        kpt = kpt_pred.view(bs, na, nkpt, ndim)

        # Decode: (pred * 2.0 + (anchor - 0.5)) * stride
        # anchor_points: (na, 2), stride_tensor: (na, 1)
        anchor_x = anchor_points[:, 0].view(1, -1, 1)  # (1, na, 1)
        anchor_y = anchor_points[:, 1].view(1, -1, 1)  # (1, na, 1)
        strides = stride_tensor.view(1, -1, 1)  # (1, na, 1)

        # Use non-inplace operations to avoid breaking gradient computation
        kpt_x = (kpt[..., 0] * 2.0 + (anchor_x - 0.5)) * strides
        kpt_y = (kpt[..., 1] * 2.0 + (anchor_y - 0.5)) * strides

        # Clamp keypoints to image bounds (non-inplace)
        if hasattr(self, 'imgsz') and self.imgsz is not None:
            kpt_x = kpt_x.clamp(0, self.imgsz[1])
            kpt_y = kpt_y.clamp(0, self.imgsz[0])

        # Stack back to (bs, na, nkpt, 2)
        kpt = torch.stack([kpt_x, kpt_y], dim=-1)

        return kpt.view(bs, na, nk)

    def calculate_keypoints_loss(self, fg_mask, target_gt_idx, keypoints, batch_idx,
                                 stride_tensor, pred_kpts):
        """Calculate keypoint loss (without bbox normalization).

        Args:
            fg_mask: foreground mask (bs, na)
            target_gt_idx: ground truth indices (bs, na) - indices into per-batch GT
            keypoints: gt keypoints (total_objects, nkpt, ndim) - all batches concatenated, in pixel coords
            batch_idx: batch indices (total_objects,) - which batch each GT belongs to
            stride_tensor: stride tensor (na, 1) - stride for each anchor
            pred_kpts: predicted keypoints (bs, na, nk)
        """
       
        loss_pose = torch.zeros(1, device=self.device)
        loss_kobj = torch.zeros(1, device=self.device)

        # if fg_mask.sum() == 0 or keypoints.numel() == 0:
        #     print(f"[DEBUG] Early return: fg_mask.sum()={fg_mask.sum()}, keypoints.numel()={keypoints.numel()}")
        #     return loss_pose, loss_kobj

        bs, na, nk = pred_kpts.shape
        nkpt = self.kpt_shape[0]
        ndim = self.kpt_shape[1]

        # Get predictions for foreground anchors
        pred_kpts_fg = pred_kpts[fg_mask]  # (fg_count, nk)
        pred_kpts_fg = pred_kpts_fg.view(-1, nkpt, ndim)  # (fg_count, nkpt, ndim)

        # Build target keypoints for each foreground anchor
        target_kpts_list = []
        stride_list = []
        areas_list = []
        batch_idx_flat = batch_idx.view(-1)

        for b in range(bs):
            # Get mask of GT objects belonging to this batch
            batch_mask = (batch_idx_flat == b)
            batch_kpts = keypoints[batch_mask]  # (num_gt_in_batch, nkpt, ndim)

            # Get foreground anchors for this batch
            fg_in_batch = fg_mask[b]  # (na,)
            gt_indices = target_gt_idx[b][fg_in_batch]  # indices into batch_kpts
            strides_in_batch = stride_tensor.view(-1)[fg_in_batch]  # strides for fg anchors

            for i, gt_idx in enumerate(gt_indices):
                gt_idx = int(gt_idx.item())
                if 0 <= gt_idx < len(batch_kpts):
                    kpts = batch_kpts[gt_idx]  # (nkpt, ndim)
                    target_kpts_list.append(kpts)
                    stride_list.append(strides_in_batch[i])

                    # Calculate area from keypoint bounding box
                    kpt_xs = kpts[:, 0]
                    kpt_ys = kpts[:, 1]
                    w = (kpt_xs.max() - kpt_xs.min()).clamp(min=1.0)
                    h = (kpt_ys.max() - kpt_ys.min()).clamp(min=1.0)
                    area = w * h
                    areas_list.append(area)

        # if len(target_kpts_list) == 0:
        #     print(f"[DEBUG] target_kpts_list is empty!")
        #     return loss_pose, loss_kobj

        target_kpts = torch.stack(target_kpts_list)  # (fg_count, nkpt, ndim)
        strides = torch.stack(stride_list).view(-1, 1, 1)  # (fg_count, 1, 1)
        areas = torch.stack(areas_list)  # (fg_count,)

       
        if pred_kpts_fg.shape[0] != target_kpts.shape[0]:
            # print(f"[DEBUG] Shape mismatch: pred={pred_kpts_fg.shape[0]}, target={target_kpts.shape[0]}")
            return loss_pose, loss_kobj

       
        kpt_mask = torch.ones_like(target_kpts[..., 0], dtype=torch.float32)  # all keypoints visible (float for loss computation)
        loss_pose = self.keypoint_loss(pred_kpts_fg, target_kpts, kpt_mask, areas.unsqueeze(-1))

        if not torch.isfinite(loss_pose):
            loss_pose = (pred_kpts_fg * 0).sum()

        
        with torch.no_grad():
            d = (pred_kpts_fg[..., :2] - target_kpts[..., :2]).pow(2).sum(dim=-1)  # squared distances per keypoint
            s = areas.unsqueeze(-1)  # (fg_count, 1)
            kpt_oks = torch.exp(-d / (2 * s * (self.keypoint_loss.sigmas ** 2) + 1e-9))  # OKS per keypoint
            oks = kpt_oks.mean(dim=-1)  # average OKS per anchor


        # Since we don't have a separate objectness prediction, use classification confidence as proxy
        loss_kobj = ((1.0 - oks) ** 2).mean() 

        return loss_pose.unsqueeze(0), loss_kobj.unsqueeze(0)



class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-obb.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model)
        # NOTE: store following info as it's changeable in __call__
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        feats = preds[1] if isinstance(preds, tuple) else preds
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion(vp_feats, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """Extract visual-prompt features from the model output."""
        vnc = feats[0].shape[1] - self.ori_reg_max * 4 - self.ori_nc

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc

        return [
            torch.cat((box, cls_vp), dim=1)
            for box, _, cls_vp in [xi.split((self.ori_reg_max * 4, self.ori_nc, vnc), dim=1) for xi in feats]
        ]


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion((vp_feats, pred_masks, proto), batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]