"""Simple viewer to browse pose predictions with q/e and exit with Esc."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview pose results and flip images with q/e, Esc to quit.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/pose/train/weights/best.pt"),
        help="Path to pose model (.pt or .yaml).",
    )
    parser.add_argument(
        "--img-dir",
        type=Path,
        default=Path("datasets"),
        help="Root directory containing images (searches recursively).",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    return parser.parse_args()


def collect_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    imgs = sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])
    if not imgs:
        raise FileNotFoundError(f"No images found under {root}")
    return imgs


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    images = collect_images(args.img_dir)
    model = YOLO(str(args.model))
    idx = 0
    win_name = "Pose Preview (q/e to navigate, Esc to exit)"

    while True:
        img_path = images[idx]
        results = model(img_path, imgsz=args.imgsz, verbose=False)[0]
        annotated = results.plot()  # returns BGR image
        cv2.putText(
            annotated,
            f"{idx + 1}/{len(images)}: {img_path.name}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(win_name, annotated)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # Esc
            break
        if key == ord("q"):
            idx = (idx - 1) % len(images)
        elif key == ord("e"):
            idx = (idx + 1) % len(images)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
