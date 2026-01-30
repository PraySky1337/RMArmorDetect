from __future__ import annotations

import argparse
import ast
import random
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils import YAML, ops

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview ONNX YOLO26 pose model.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/pose/train/weights/best.onnx"),
        help="Path to pose ONNX model.",
    )
    parser.add_argument(
        "--img-dir",
        type=Path,
        default=Path("datasets/images"),
        help="Directory containing images.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution provider: auto/cpu/cuda.",
    )
    parser.add_argument(
        "--box-format",
        choices=("auto", "xywh", "xyxy"),
        default="auto",
        help="Box format of model output.",
    )
    parser.add_argument("--data", type=Path, default=Path("data.yaml"), help="Optional data.yaml for class names.")
    return parser.parse_args()


def collect_images(root: Path) -> list[Path]:
    imgs = sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    if not imgs:
        raise FileNotFoundError(f"No images found under {root}")
    return imgs


def get_providers(device: str) -> list[str]:
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # auto
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def parse_metadata(meta: dict[str, str]) -> tuple[tuple[int, int] | None, list[str] | None]:
    kpt_shape = None
    names = None
    if "kpt_shape" in meta:
        try:
            kpt_shape = tuple(ast.literal_eval(meta["kpt_shape"]))  # type: ignore[arg-type]
        except Exception:
            kpt_shape = None
    if "names" in meta:
        try:
            name_obj = ast.literal_eval(meta["names"])
            if isinstance(name_obj, dict):
                names = [name_obj[k] for k in sorted(name_obj)]
            elif isinstance(name_obj, (list, tuple)):
                names = list(name_obj)
        except Exception:
            names = None
    return kpt_shape, names


def load_names_from_data(data_path: Path) -> list[str] | None:
    if not data_path.exists():
        return None
    data = YAML.load(data_path)
    if isinstance(data, dict) and "names" in data:
        names = data["names"]
        if isinstance(names, dict):
            return [names[k] for k in sorted(names)]
        if isinstance(names, list):
            return names
    return None


def preprocess_image(img_path: Path, imgsz: int):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {img_path}")

    orig_shape = img_bgr.shape[:2]
    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_letterboxed = letterbox(image=img_rgb)

    img_tensor = img_letterboxed.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_tensor = np.expand_dims(img_tensor, 0)

    ratio = min(imgsz / orig_shape[0], imgsz / orig_shape[1])
    new_unpad = round(orig_shape[1] * ratio), round(orig_shape[0] * ratio)
    pad_w = (imgsz - new_unpad[0]) / 2
    pad_h = (imgsz - new_unpad[1]) / 2
    ratio_pad = ((ratio, ratio), (pad_w, pad_h))

    return img_bgr, img_tensor, orig_shape, ratio_pad


def _resolve_box_format(boxes: torch.Tensor, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    invalid = (boxes[:, 2] < boxes[:, 0]) | (boxes[:, 3] < boxes[:, 1])
    if invalid.float().mean().item() > 0.2:
        return "xywh"
    return "xyxy"


def postprocess(preds, orig_shape, ratio_pad, conf_thres, imgsz, kpt_shape, box_format):
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    preds = torch.from_numpy(preds) if isinstance(preds, np.ndarray) else preds
    preds = preds[0]

    boxes = preds[:, :4]
    conf = preds[:, 4]
    cls_idx = preds[:, 5].long()
    kpts = preds[:, 6:]

    nkpt, ndim = kpt_shape
    kpts = kpts.view(-1, nkpt, ndim)

    box_format = _resolve_box_format(boxes, box_format)
    if box_format == "xywh":
        xyxy_boxes = torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:]], dim=1)
    else:
        xyxy_boxes = boxes

    valid_mask = conf > conf_thres
    if valid_mask.sum() == 0:
        return []

    valid_boxes = xyxy_boxes[valid_mask]
    valid_conf = conf[valid_mask]
    valid_cls = cls_idx[valid_mask]
    valid_kpts = kpts[valid_mask]

    scaled_boxes = ops.scale_coords((imgsz, imgsz), valid_boxes, orig_shape, ratio_pad=ratio_pad)
    scaled_kpts = ops.scale_coords((imgsz, imgsz), valid_kpts, orig_shape, ratio_pad=ratio_pad)

    orig_h, orig_w = orig_shape
    results = []
    for i in range(len(scaled_boxes)):
        kpt = scaled_kpts[i].cpu().numpy()
        if (kpt[..., 0] >= 0).all() and (kpt[..., 0] < orig_w).all() and (kpt[..., 1] >= 0).all() and (
            kpt[..., 1] < orig_h
        ).all():
            results.append(
                {
                    "keypoints": kpt,
                    "conf": float(valid_conf[i].item()),
                    "cls": int(valid_cls[i].item()),
                    "box": scaled_boxes[i].cpu().numpy(),
                }
            )

    return results


def color_from_name(name: str) -> tuple[int, int, int]:
    if not name:
        return (0, 255, 0)
    key = name[0].upper()
    if key == "B":
        return (255, 0, 0)
    if key == "R":
        return (0, 0, 255)
    if key == "G":
        return (128, 128, 128)
    if key == "P":
        return (255, 0, 255)
    return (0, 255, 0)


def draw_detections(img, detections, class_names: list[str]):
    for det in detections:
        kpts = det["keypoints"].astype(np.int32)
        conf = det["conf"]
        cls_idx = det["cls"]

        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else "?"
        bgr_color = color_from_name(cls_name)

        pts = kpts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=bgr_color, thickness=2)
        for pt in kpts:
            cv2.circle(img, tuple(pt), 3, bgr_color, -1)
        label = f"{cls_name} {conf:.2f}"
        x, y = kpts[0]
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2, cv2.LINE_AA)
    return img


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    images = collect_images(args.img_dir)
    providers = get_providers(args.device)
    session = ort.InferenceSession(str(args.model), providers=providers)
    input_name = session.get_inputs()[0].name

    meta = session.get_modelmeta().custom_metadata_map
    kpt_shape, names = parse_metadata(meta)
    if names is None:
        names = load_names_from_data(args.data) or []
    if kpt_shape is None:
        kpt_shape = (4, 2)

    idx = 0
    conf_thres = args.conf
    win_name = "ONNX Pose Preview (q/e: nav, w/s: conf, r: random, Esc: exit)"

    while True:
        img_path = images[idx]
        img_bgr, img_tensor, orig_shape, ratio_pad = preprocess_image(img_path, args.imgsz)

        preds = session.run(None, {input_name: img_tensor})[0]
        detections = postprocess(
            preds,
            orig_shape,
            ratio_pad,
            conf_thres,
            args.imgsz,
            kpt_shape,
            args.box_format,
        )

        annotated = img_bgr.copy()
        annotated = draw_detections(annotated, detections, names)

        info_lines = [
            f"{idx + 1}/{len(images)}: {img_path.name}",
            f"conf: {conf_thres:.2f} | detections: {len(detections)}",
            f"provider: {providers[0]}",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(annotated, line, (10, 25 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(win_name, annotated)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # Esc
            break
        if key == ord("q"):
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
