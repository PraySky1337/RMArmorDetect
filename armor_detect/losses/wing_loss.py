"""
Wing Loss for robust keypoint regression.

Reference: Wing Loss for Robust Facial Landmark Localisation with CNNs
https://arxiv.org/abs/1711.06753

Wing Loss uses a logarithmic function for small errors (more sensitive)
and a linear function for large errors (more robust to outliers).
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class WingLoss(nn.Module):
    """Wing Loss for robust keypoint/landmark regression.

    Wing Loss is designed to be more sensitive to small localization errors
    while remaining robust to large errors (outliers). It uses:
    - Logarithmic part for |x| < omega: omega * ln(1 + |x|/epsilon)
    - Linear part for |x| >= omega: |x| - C

    where C = omega - omega * ln(1 + omega/epsilon) ensures continuity.

    Args:
        omega (float): Threshold between logarithmic and linear region.
            Larger omega means more errors are treated with log (more sensitive).
            Default: 10.0
        epsilon (float): Smoothing parameter for the logarithmic part.
            Smaller epsilon means steeper gradient for small errors.
            Default: 2.0

    Example:
        >>> loss_fn = WingLoss(omega=10.0, epsilon=2.0)
        >>> pred = torch.randn(16, 8)  # 16 samples, 8 keypoint coords
        >>> target = torch.randn(16, 8)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(self, omega: float = 10.0, epsilon: float = 2.0):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        # Pre-compute constant C for continuity at omega
        self.C = omega - omega * math.log(1 + omega / epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Wing Loss between predictions and targets.

        Args:
            pred (torch.Tensor): Predicted values, shape (N, D) or any broadcastable shape
            target (torch.Tensor): Ground truth values, same shape as pred

        Returns:
            torch.Tensor: Scalar loss value (mean over all elements)
        """
        delta_y = (target - pred).abs()

        # Split into small errors (logarithmic) and large errors (linear)
        small_error_mask = delta_y < self.omega

        # Logarithmic loss for small errors: omega * ln(1 + |x|/epsilon)
        small_errors = delta_y[small_error_mask]
        loss_small = self.omega * torch.log(1.0 + small_errors / self.epsilon)

        # Linear loss for large errors: |x| - C
        large_errors = delta_y[~small_error_mask]
        loss_large = large_errors - self.C

        # Combine and normalize
        total_count = len(small_errors) + len(large_errors)
        if total_count == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        total_loss = loss_small.sum() + loss_large.sum()
        return total_loss / (total_count + 1e-9)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(omega={self.omega}, epsilon={self.epsilon})"


class WingLossWithMask(WingLoss):
    """Wing Loss with keypoint visibility mask support.

    Extends WingLoss to handle cases where some keypoints may be invisible
    or invalid, using a binary mask to exclude them from loss computation.

    Args:
        omega (float): Threshold between logarithmic and linear region. Default: 10.0
        epsilon (float): Smoothing parameter. Default: 2.0

    Example:
        >>> loss_fn = WingLossWithMask()
        >>> pred = torch.randn(16, 4, 2)  # 16 samples, 4 keypoints, 2D
        >>> target = torch.randn(16, 4, 2)
        >>> mask = torch.ones(16, 4)  # All keypoints visible
        >>> mask[0, 2] = 0  # Hide keypoint 2 in sample 0
        >>> loss = loss_fn(pred, target, mask)
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute masked Wing Loss.

        Args:
            pred (torch.Tensor): Predicted keypoints, shape (N, K, 2) or (N, K*2)
            target (torch.Tensor): Ground truth keypoints, same shape as pred
            mask (torch.Tensor, optional): Visibility mask, shape (N, K).
                1 = visible (include in loss), 0 = invisible (exclude).
                If None, all keypoints are considered visible.

        Returns:
            torch.Tensor: Scalar loss value
        """
        if mask is None:
            return super().forward(pred, target)

        # Flatten if needed: (N, K, 2) -> (N, K*2)
        if pred.dim() == 3:
            N, K, D = pred.shape
            pred = pred.reshape(N, K * D)
            target = target.reshape(N, K * D)
            # Expand mask to match: (N, K) -> (N, K*D)
            mask = mask.unsqueeze(-1).expand(N, K, D).reshape(N, K * D)

        # Apply mask
        pred_masked = pred[mask > 0]
        target_masked = target[mask > 0]

        if pred_masked.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return super().forward(pred_masked, target_masked)
