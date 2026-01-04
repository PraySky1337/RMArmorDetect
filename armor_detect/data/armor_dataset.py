"""
Armor Dataset - Dataset class for armor detection with color and size field support.

This module provides the ArmorDataset class that wraps YOLODataset
and ensures proper handling of the color and size classification fields.
"""

import numpy as np
from typing import Any

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER


class ArmorDataset(YOLODataset):
    """Dataset class for armor detection with color and size field support.

    This class extends YOLODataset to handle the triple classification format
    used in armor detection (number class + color class + size class).

    Label format: color size cls x1 y1 x2 y2 x3 y3 x4 y4
    where:
    - color: color class (0-3: B, R, N, P)
    - size: size class (0-1: s, b)
    - cls: number class (0-7: G, 1, 2, 3, 4, 5, O, B)
    - x1-x4, y1-y4: 4 corner keypoints (normalized coordinates)

    Args:
        *args: Arguments passed to YOLODataset.
        **kwargs: Keyword arguments passed to YOLODataset.
    """

    def __init__(self, *args, **kwargs):
        """Initialize ArmorDataset.

        Ensures task is set to 'pose' for keypoint detection.
        """
        kwargs.setdefault('task', 'pose')
        super().__init__(*args, **kwargs)

    def cache_labels(self, path: str = "./labels.cache") -> dict:
        """Cache labels with color and size field validation.

        Args:
            path: Path to cache file.

        Returns:
            dict: Cached labels with validated color and size fields.
        """
        labels = super().cache_labels(path)

        # Ensure color and size fields exist and are properly formatted
        for lb in labels:
            self._validate_color_field(lb)
            self._validate_size_field(lb)

        return labels

    def get_labels(self) -> list[dict[str, Any]]:
        """Get labels with color and size field validation.

        Returns:
            list: List of label dictionaries with validated color and size fields.
        """
        labels = super().get_labels()

        # Ensure color and size fields exist and are properly formatted
        for lb in labels:
            self._validate_color_field(lb)
            self._validate_size_field(lb)

        return labels

    def _validate_color_field(self, label: dict) -> None:
        """Validate and fix color field in a label dictionary.

        Args:
            label: Label dictionary to validate.
        """
        color_arr = label.get("color", None)

        if color_arr is None:
            # No color field - create default (zeros)
            num_objects = len(label.get("cls", []))
            label["color"] = np.zeros((num_objects,), dtype=np.int64)
        else:
            # Ensure color is 1D array of int64
            color_arr = np.asarray(color_arr, dtype=np.int64)
            if color_arr.ndim == 2:
                color_arr = color_arr.squeeze(-1)  # (n, 1) -> (n,)
            label["color"] = color_arr.flatten()

    def _validate_size_field(self, label: dict) -> None:
        """Validate and fix size field in a label dictionary.

        Args:
            label: Label dictionary to validate.
        """
        size_arr = label.get("size", None)

        if size_arr is None:
            # No size field - create default (zeros)
            num_objects = len(label.get("cls", []))
            label["size"] = np.zeros((num_objects,), dtype=np.int64)
        else:
            # Ensure size is 1D array of int64
            size_arr = np.asarray(size_arr, dtype=np.int64)
            if size_arr.ndim == 2:
                size_arr = size_arr.squeeze(-1)  # (n, 1) -> (n,)
            label["size"] = size_arr.flatten()

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Collate batch with color and size field support.

        Args:
            batch: List of sample dictionaries.

        Returns:
            dict: Collated batch dictionary.
        """
        import torch

        # Use parent collate function
        collated = YOLODataset.collate_fn(batch)

        # Handle color field if present
        if batch and "color" in batch[0]:
            colors = []
            for sample in batch:
                if "color" in sample:
                    color = sample["color"]
                    if isinstance(color, np.ndarray):
                        color = torch.from_numpy(color)
                    colors.append(color)

            if colors:
                collated["color"] = torch.cat(colors, 0)

        # Handle size field if present
        if batch and "size" in batch[0]:
            sizes = []
            for sample in batch:
                if "size" in sample:
                    size = sample["size"]
                    if isinstance(size, np.ndarray):
                        size = torch.from_numpy(size)
                    sizes.append(size)

            if sizes:
                collated["size"] = torch.cat(sizes, 0)

        return collated


def build_armor_dataloader(
    dataset_path: str,
    batch_size: int,
    img_size: int = 640,
    augment: bool = True,
    workers: int = 8,
    shuffle: bool = True,
    **kwargs,
) -> Any:
    """Build a dataloader for armor detection.

    Args:
        dataset_path: Path to dataset YAML file.
        batch_size: Batch size.
        img_size: Input image size.
        augment: Whether to apply augmentation.
        workers: Number of dataloader workers.
        shuffle: Whether to shuffle data.
        **kwargs: Additional arguments for dataloader.

    Returns:
        DataLoader for armor detection.
    """
    from ultralytics.data import build_dataloader

    dataset = ArmorDataset(
        img_path=dataset_path,
        imgsz=img_size,
        augment=augment,
        task='pose',
        **kwargs,
    )

    return build_dataloader(
        dataset,
        batch=batch_size,
        workers=workers,
        shuffle=shuffle,
    )
