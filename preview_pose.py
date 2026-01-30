# YOLO26 end-to-end pose preview script.
#
# 快捷键:
#   q/e  - 上一张/下一张图片
#   w/s  - 增加/减少置信度阈值 (±0.05)
#   r    - 随机图片
#   Esc  - 退出

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops

# 类别名称 (从 data.yaml 中读取)
# 格式: {颜色}{大小}{数字} - B/R/G/P = Blue/Red/Gray/Purple, s/b = small/big, 0-7 = 数字
CLASS_NAMES = [
    "Bs0", "Bs1", "Bs2", "Bs3", "Bs4", "Bs5", "Bs6", "Bs7",
    "Bb0", "Bb1", "Bb2", "Bb3", "Bb4", "Bb5", "Bb6", "Bb7",
    "Rs0", "Rs1", "Rs2", "Rs3", "Rs4", "Rs5", "Rs6", "Rs7",
    "Rb0", "Rb1", "Rb2", "Rb3", "Rb4", "Rb5", "Rb6", "Rb7",
    "Gs0", "Gs1", "Gs2", "Gs3", "Gs4", "Gs5", "Gs6", "Gs7",
    "Gb0", "Gb1", "Gb2", "Gb3", "Gb4", "Gb5", "Gb6", "Gb7",
    "Ps0", "Ps1", "Ps2", "Ps3", "Ps4", "Ps5", "Ps6", "Ps7",
    "Pb0", "Pb1", "Pb2", "Pb3", "Pb4", "Pb5", "Pb6", "Pb7",
]

# 颜色映射 (BGR) - 根据类别名称的第一个字母 (颜色) 分配
COLOR_MAP = {
    "B": (255, 0, 0),    # Blue
    "R": (0, 0, 255),    # Red
    "G": (128, 128, 128),# Gray
    "P": (255, 0, 255),  # Purple
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO26 end-to-end pose preview")
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
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    return parser.parse_args()


def collect_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    imgs = sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])
    if not imgs:
        raise FileNotFoundError(f"No images found under {root}")
    return imgs


def preprocess_image(img_path: Path, imgsz: int, device: torch.device):
    """Load and preprocess image for inference using Ultralytics standard LetterBox."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {img_path}")

    orig_shape = img_bgr.shape[:2]  # (h, w)

    # Use Ultralytics standard LetterBox for preprocessing
    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_letterboxed = letterbox(image=img_rgb)

    # To tensor (BCHW format)
    img_tensor = torch.from_numpy(img_letterboxed).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Get ratio_pad for postprocessing (gain, (pad_w, pad_h))
    ratio = min(imgsz / orig_shape[0], imgsz / orig_shape[1])
    new_unpad = round(orig_shape[1] * ratio), round(orig_shape[0] * ratio)
    pad_w = (imgsz - new_unpad[0]) / 2
    pad_h = (imgsz - new_unpad[1]) / 2
    ratio_pad = ((ratio, ratio), (pad_w, pad_h))

    return img_bgr, img_tensor, orig_shape, ratio_pad


def postprocess(preds, orig_shape, ratio_pad, conf_thres, model, imgsz):
    """Parse YOLO26 end-to-end model output.

    YOLO26 end-to-end model output format:
    - Returns tuple: (processed_output, raw_predictions)
    - Processed output: (batch_size, max_det, 6 + nk)
      Format: [x, y, w, h, max_class_prob, class_index, keypoints...]
    """
    if isinstance(preds, (tuple, list)):
        # YOLO26 returns (processed_output, raw_predictions) tuple
        preds = preds[0]

    # preds shape: (1, max_det, 6 + nk) for batch_size=1
    # Format: [x, y, w, h, max_class_prob, class_index, kpts...]
    preds = preds[0]  # Remove batch dimension

    # Extract components
    boxes = preds[:, :4]  # (max_det, 4) - [x, y, w, h] in xywh format
    cls_idx = preds[:, 5].long()  # (max_det,) - class indices
    conf = preds[:, 4]  # (max_det,) - confidence scores
    kpts = preds[:, 6:]  # (max_det, nk) - keypoints flattened

    nkpt = model.model[-1].kpt_shape[0]
    ndim = model.model[-1].kpt_shape[1]

    # Reshape keypoints: (max_det, nkpt * ndim) -> (max_det, nkpt, ndim)
    kpts = kpts.view(-1, nkpt, ndim)

    # Confidence filter
    valid_mask = conf > conf_thres
    if valid_mask.sum() == 0:
        return []

    valid_boxes = boxes[valid_mask]
    valid_conf = conf[valid_mask]
    valid_cls = cls_idx[valid_mask]
    valid_kpts = kpts[valid_mask]

    # Convert xywh to xyxy for ops.scale_coords
    xyxy_boxes = torch.cat([
        valid_boxes[:, :2],  # xy
        valid_boxes[:, :2] + valid_boxes[:, 2:]  # xy + wh = x2y2
    ], dim=1)

    # Scale boxes to original image
    scaled_boxes = ops.scale_coords((imgsz, imgsz), xyxy_boxes, orig_shape, ratio_pad=ratio_pad)

    # Scale keypoints to original image
    scaled_kpts = ops.scale_coords((imgsz, imgsz), valid_kpts, orig_shape, ratio_pad=ratio_pad)

    # Convert results to list
    orig_h, orig_w = orig_shape
    results = []
    for i in range(len(scaled_boxes)):
        kpt = scaled_kpts[i].cpu().numpy()
        box = scaled_boxes[i].cpu().numpy()
        # Check if keypoints are within image bounds
        if (kpt[..., 0] >= 0).all() and (kpt[..., 0] < orig_w).all() and \
           (kpt[..., 1] >= 0).all() and (kpt[..., 1] < orig_h).all():
            results.append({
                'keypoints': kpt,
                'conf': valid_conf[i].item(),
                'cls': int(valid_cls[i].item()),
                'box': box,
            })

    return results


def draw_detections(img, detections):
    """Draw keypoints, polygons, and labels on image."""
    for det in detections:
        kpts = det['keypoints'].astype(np.int32)
        conf = det['conf']
        cls_idx = det['cls']

        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else "?"
        # 根据类别名称的第一个字母获取颜色
        color_key = cls_name[0] if cls_name else "?"
        bgr_color = COLOR_MAP.get(color_key, (0, 255, 0))  # 默认绿色

        # Draw polygon
        pts = kpts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=bgr_color, thickness=2)

        # Draw keypoints
        for pt in kpts:
            cv2.circle(img, tuple(pt), 3, bgr_color, -1)

        # Draw label
        label = f"{cls_name} {conf:.2f}"
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

    win_name = "Pose Preview (q/e: nav, w/s: conf, r: random, Esc: exit)"

    while True:
        img_path = images[idx]

        # Inference using YOLO model's forward pass
        img_bgr, img_tensor, orig_shape, ratio_pad = preprocess_image(
            img_path, args.imgsz, device
        )

        with torch.no_grad():
            # YOLO26 end-to-end model forward
            preds = model.model(img_tensor)

        detections = postprocess(
            preds, orig_shape, ratio_pad,
            conf_thres, model.model, args.imgsz
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
            idx = random.randint(0, len(images) - 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
