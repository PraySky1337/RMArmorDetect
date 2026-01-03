"""
Armor Pose Head - Detection head for armor detection with dual classification branches.

This module provides the ArmorPoseHead class for keypoint-based armor detection
without bounding boxes, using dual classification branches for number and color.
"""

import math
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.utils.tal import make_anchors
from ultralytics.utils import NOT_MACOS14

from armor_detect.utils import INITIAL_CONF


class ArmorPoseHead(nn.Module):
    """YOLO Pose head for armor detection WITHOUT bounding box.

    This class predicts keypoints and two classification branches for armor detection:
    - Number class (1-8): predicted by cv3 branch
    - Color class (R, B, N, P): predicted by cv_color branch
    - Keypoints: 4 corner points with 2D coordinates (defines object location)

    NO bounding box prediction - keypoints alone define the armor location.

    The label format is: color cls x1 y1 x2 y2 x3 y3 x4 y4
    During inference, the two class branches are merged into joint predictions.

    Attributes:
        nc_num (int): Number of number classes (1-8).
        nc_color (int): Number of color classes (R, B, N, P).
        kpt_shape (tuple): Number of keypoints and dimensions (4, 2 for 4 corners).
        nk (int): Total number of keypoint values.
        nl (int): Number of detection layers.
        cv3 (nn.ModuleList): Convolution layers for number classification.
        cv_color (nn.ModuleList): Convolution layers for color classification.
        cv4 (nn.ModuleList): Convolution layers for keypoint prediction.
    """

    dynamic = False  # Force grid reconstruction
    export = False  # Export mode
    format = None  # Export format
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(
        self,
        nc_num: int = 8,
        nc_color: int = 4,
        kpt_shape: tuple = (4, 2),
        ch: tuple = (),
    ):
        """Initialize ArmorPoseHead with dual classification branches and keypoint prediction.

        Args:
            nc_num: Number of number classes (1-8 for armor numbers).
            nc_color: Number of color classes (R, B, N, P).
            kpt_shape: Number of keypoints and dimensions (4, 2 for x,y coordinates).
            ch: Tuple of channel sizes from backbone feature maps.
        """
        super().__init__()

        if not ch or len(ch) == 0:
            ch = (256, 512, 1024)

        self.nc_num = nc_num
        self.nc_color = nc_color
        self.nc = nc_num  # For compatibility with base detection classes
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]  # 8 for 4 points x 2D
        self.nl = len(ch)  # Number of detection layers
        self.no = nc_num + nc_color + self.nk  # Outputs per anchor (no bbox)

        # Stride will be computed dynamically during first forward pass
        self.stride = torch.zeros(self.nl)
        self._stride_initialized = False

        # Number classification branch
        c3_num = max(ch[0], min(self.nc_num, 100))
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3_num, 1)),
                nn.Sequential(DWConv(c3_num, c3_num, 3), Conv(c3_num, c3_num, 1)),
                nn.Conv2d(c3_num, self.nc_num, 1),
            )
            for x in ch
        )

        # Color classification branch
        c3_color = max(ch[0], min(self.nc_color, 100))
        self.cv_color = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3_color, 1)),
                nn.Sequential(DWConv(c3_color, c3_color, 3), Conv(c3_color, c3_color, 1)),
                nn.Conv2d(c3_color, self.nc_color, 1),
            )
            for x in ch
        )

        # Keypoint prediction branch
        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1))
            for x in ch
        )

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Perform forward pass with dual classification branches and keypoint prediction.

        Args:
            x: List of feature maps from backbone/neck.

        Returns:
            During training: (feats, (cls_num, cls_color, kpt))
            During inference: (combined_output, (feats, (cls_num, cls_color, kpt)))
            During export: combined tensor
        """
        bs = x[0].shape[0]

        # Initialize stride on first forward pass
        if not self._stride_initialized:
            self._init_stride(x)

        # Extract predictions from each branch
        kpt = torch.cat(
            [self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1
        )
        cls_num = torch.cat(
            [self.cv3[i](x[i]).view(bs, self.nc_num, -1) for i in range(self.nl)], -1
        )
        cls_color = torch.cat(
            [self.cv_color[i](x[i]).view(bs, self.nc_color, -1) for i in range(self.nl)], -1
        )

        # Save original features for loss computation
        feats = [xi.clone() for xi in x]

        # Training mode
        if self.training:
            return feats, (cls_num, cls_color, kpt)

        # Inference mode - initialize anchors and strides
        shape = x[0].shape  # BCHW
        if self.dynamic or self.shape != shape:
            anchors, strides = make_anchors(x, self.stride, 0.5)
            self.anchors = anchors.transpose(0, 1)  # (2, na)
            self.strides = strides.squeeze(-1)  # (na,)
            self.shape = shape

        # Decode keypoints
        pred_kpt = self._kpts_decode(bs, kpt)

        # Get classification predictions
        cls_num_sig = cls_num.sigmoid()
        cls_color_sig = cls_color.sigmoid()

        if self.export:
            # Export mode: merge into joint class + keypoints
            joint_cls = self._merge_branches(cls_num, cls_color)
            joint_cls = joint_cls.permute(0, 2, 1)
            return torch.cat([joint_cls, pred_kpt], 1)

        # Validation mode: output separately
        combined = torch.cat([cls_num_sig, cls_color_sig, pred_kpt], 1)
        return (combined, (feats, (cls_num, cls_color, kpt)))

    def _init_stride(self, x: list[torch.Tensor]) -> None:
        """Initialize stride from feature map sizes.

        Args:
            x: List of feature maps from backbone/neck.
        """
        # Assume input image size is the largest feature map size * 8 (P3 stride)
        # Common strides: P3=8, P4=16, P5=32
        base_stride = 8
        self.stride = torch.tensor(
            [base_stride * (2**i) for i in range(self.nl)],
            device=x[0].device,
            dtype=x[0].dtype,
        )
        self._stride_initialized = True

    def _merge_branches(
        self, cls_num: torch.Tensor, cls_color: torch.Tensor
    ) -> torch.Tensor:
        """Merge number and color classification outputs into joint predictions.

        Args:
            cls_num: Number classification logits (bs, nc_num, na)
            cls_color: Color classification logits (bs, nc_color, na)

        Returns:
            Joint probability matrix (bs, na, nc_color * nc_num)
        """
        prob_num = cls_num.sigmoid().permute(0, 2, 1)  # (bs, na, nc_num)
        prob_color = cls_color.sigmoid().permute(0, 2, 1)  # (bs, na, nc_color)

        bs, na, _ = prob_num.shape
        prob_color_flat = prob_color.reshape(bs * na, self.nc_color, 1)
        prob_num_flat = prob_num.reshape(bs * na, 1, self.nc_num)
        joint_prob = torch.bmm(prob_color_flat, prob_num_flat)
        return joint_prob.view(bs, na, self.nc_color * self.nc_num)

    def _kpts_decode(self, bs: int, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from raw predictions to pixel coordinates.

        Args:
            bs: Batch size
            kpts: Raw keypoint predictions (bs, nk, na)

        Returns:
            Decoded keypoints in pixel coordinates (bs, nk, na)
        """
        ndim = self.kpt_shape[1]

        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)

        y = kpts.clone()

        # Handle visibility dimension if present
        if ndim == 3:
            if NOT_MACOS14:
                y[:, 2::ndim].sigmoid_()
            else:
                y[:, 2::ndim] = y[:, 2::ndim].sigmoid()

        # Decode x, y coordinates
        anchor_x = self.anchors[0].unsqueeze(0).unsqueeze(0)  # (1, 1, na)
        anchor_y = self.anchors[1].unsqueeze(0).unsqueeze(0)  # (1, 1, na)
        stride_expanded = self.strides.unsqueeze(0).unsqueeze(0)  # (1, 1, na)

        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (anchor_x - 0.5)) * stride_expanded
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (anchor_y - 0.5)) * stride_expanded

        # Clamp to image bounds
        if self.shape is not None:
            imgsz_h = self.shape[2] * self.stride[0]
            imgsz_w = self.shape[3] * self.stride[0]
        else:
            max_anchor_y = self.anchors[1].max().item()
            max_anchor_x = self.anchors[0].max().item()
            max_stride = self.strides.max().item()
            imgsz_h = (max_anchor_y + 1) * max_stride
            imgsz_w = (max_anchor_x + 1) * max_stride

        y[:, 0::ndim] = y[:, 0::ndim].clamp(0, imgsz_w)
        y[:, 1::ndim] = y[:, 1::ndim].clamp(0, imgsz_h)

        return y

    def bias_init(self) -> None:
        """Initialize biases for classification heads.

        Sets initial confidence to approximately 1% to prevent early training instability.
        """
        initial_bias = math.log(INITIAL_CONF / (1 - INITIAL_CONF))
        for a, b in zip(self.cv3, self.cv_color):
            a[-1].bias.data[:] = initial_bias
            b[-1].bias.data[:] = initial_bias


# Alias for backward compatibility
Pose = ArmorPoseHead
