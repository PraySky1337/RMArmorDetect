"""
Enhanced pose preview with cls/color display and dynamic conf adjustment.

快捷键:
  q/e  - 上一张/下一张图片
  w/s  - 增加/减少置信度阈值 (±0.05)
  r    - 重置置信度为默认值
  Esc  - 退出
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms as torchvision_nms
from ultralytics import YOLO
from ultralytics.utils import ops


COLOR_NAMES = ["B", "R", "N", "P"]
CLASS_NAMES = ["Gs", "1b", "2s", "3s", "4s", "Os", "Bs", "Bb"]

# 颜色映射 (BGR)
COLORS = {
    "B": (255, 0, 0),    # Blue
    "R": (0, 0, 255),    # Red
    "N": (128, 128, 128),# Gray
    "P": (255, 0, 255),  # Purple
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced pose preview with cls/color info")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/pose/train/weights/best.pt"),
        help="Path to pose model (.pt).",
    )
    parser.add_argument(
        "--img-dir",
        type=Path,
        default=Path("datasets/images"),
        help="Directory containing images.",
    )
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Initial confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold.")
    return parser.parse_args()


def collect_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    imgs = sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])
    if not imgs:
        raise FileNotFoundError(f"No images found under {root}")
    return imgs


def preprocess_image(img_path: Path, imgsz: int, device: torch.device):
    """Load and preprocess image for inference."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {img_path}")

    orig_shape = img_bgr.shape[:2]  # (h, w)

    # Letterbox resize
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    new_h, new_w = int(h * scale), int(w * scale)

    img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    pad_h = (imgsz - new_h) // 2
    pad_w = (imgsz - new_w) // 2
    img_padded = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    img_padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = img_resized

    # To tensor
    img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    return img_bgr, img_tensor, orig_shape, (scale, pad_h, pad_w)


def postprocess(preds, orig_shape, preprocess_info, conf_thres, iou_thres, model):
    """Parse raw model output to get detections with cls, color, conf, keypoints."""
    if isinstance(preds, (tuple, list)):
        preds = preds[0]

    pred_data = preds[0]  # (nc_num + nc_color + nk, na)

    nc_num = model.model[-1].nc_num
    nc_color = model.model[-1].nc_color
    nkpt, ndim = model.model[-1].kpt_shape
    nk = nkpt * ndim

    # Split predictions
    num_scores = pred_data[:nc_num]  # (nc_num, na)
    color_scores = pred_data[nc_num:nc_num + nc_color]  # (nc_color, na)
    kpt_data = pred_data[nc_num + nc_color:]  # (nk, na)

    # Get confidence
    num_conf, cls_indices = num_scores.max(dim=0)
    color_indices = color_scores.argmax(dim=0)

    # Confidence filter
    valid_mask = num_conf > conf_thres
    if valid_mask.sum() == 0:
        return []

    valid_conf = num_conf[valid_mask]
    valid_cls = cls_indices[valid_mask]
    valid_color = color_indices[valid_mask]
    valid_kpts = kpt_data[:, valid_mask]

    # Reshape keypoints
    num_valid = valid_kpts.shape[1]
    kpts = valid_kpts.view(nkpt, ndim, num_valid).permute(2, 0, 1)  # (num_valid, nkpt, ndim)

    # NMS based on keypoint bounding box
    x_min = kpts[..., 0].min(dim=1).values
    y_min = kpts[..., 1].min(dim=1).values
    x_max = kpts[..., 0].max(dim=1).values
    y_max = kpts[..., 1].max(dim=1).values
    boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)

    keep = torchvision_nms(boxes.float(), valid_conf.float(), iou_thres)
    keep = keep[:100]  # max detections

    # Scale keypoints to original image
    scale, pad_h, pad_w = preprocess_info
    final_kpts = kpts[keep].cpu().numpy()
    final_kpts[..., 0] = (final_kpts[..., 0] - pad_w) / scale
    final_kpts[..., 1] = (final_kpts[..., 1] - pad_h) / scale

    final_conf = valid_conf[keep].cpu().numpy()
    final_cls = valid_cls[keep].cpu().numpy()
    final_color = valid_color[keep].cpu().numpy()

    # Filter out-of-bounds
    img_h, img_w = orig_shape
    results = []
    for i in range(len(final_kpts)):
        kpt = final_kpts[i]
        if (kpt[..., 0] >= 0).all() and (kpt[..., 0] < img_w).all() and \
           (kpt[..., 1] >= 0).all() and (kpt[..., 1] < img_h).all():
            results.append({
                'keypoints': kpt,
                'conf': final_conf[i],
                'cls': int(final_cls[i]),
                'color': int(final_color[i]),
            })

    return results


def draw_detections(img, detections):
    """Draw keypoints, polygons, and labels on image."""
    for det in detections:
        kpts = det['keypoints'].astype(np.int32)
        conf = det['conf']
        cls_idx = det['cls']
        color_idx = det['color']

        color_name = COLOR_NAMES[color_idx] if color_idx < len(COLOR_NAMES) else "?"
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else "?"
        bgr_color = COLORS.get(color_name, (0, 255, 0))

        # Draw polygon
        pts = kpts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=bgr_color, thickness=2)

        # Draw keypoints
        for pt in kpts:
            cv2.circle(img, tuple(pt), 3, bgr_color, -1)

        # Draw label
        label = f"{color_name}{cls_name} {conf:.2f}"
        x, y = kpts[0]
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, bgr_color, 2, cv2.LINE_AA)

    return img


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    images = collect_images(args.img_dir)
    model = YOLO(str(args.model))
    device = next(model.model.parameters()).device

    idx = 0
    conf_thres = args.conf
    default_conf = args.conf

    win_name = "Pose Preview (q/e: nav, w/s: conf, r: reset, Esc: exit)"

    while True:
        img_path = images[idx]

        # Inference
        img_bgr, img_tensor, orig_shape, preprocess_info = preprocess_image(
            img_path, args.imgsz, device
        )

        with torch.no_grad():
            preds = model.model(img_tensor)

        detections = postprocess(
            preds, orig_shape, preprocess_info,
            conf_thres, args.iou, model.model
        )

        # Draw
        annotated = img_bgr.copy()
        annotated = draw_detections(annotated, detections)

        # Draw info overlay
        info_lines = [
            f"{idx + 1}/{len(images)}: {img_path.name}",
            f"conf: {conf_thres:.2f} | detections: {len(detections)}",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(annotated, line, (10, 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(win_name, annotated)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # Esc
            break
        elif key == ord("q"):
            idx = (idx - 1) % len(images)
        elif key == ord("e"):
            idx = (idx + 1) % len(images)
        elif key == ord("w"):
            conf_thres = min(0.95, conf_thres + 0.05)
        elif key == ord("s"):
            conf_thres = max(0.05, conf_thres - 0.05)
        elif key == ord("r"):
            conf_thres = default_conf

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
