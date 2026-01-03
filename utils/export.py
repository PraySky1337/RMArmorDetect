"""Export dual-branch Pose model (no bbox) to ONNX for OpenVINO deployment."""

import argparse
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Export Pose model to ONNX")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/weights/best.pt"),
        help="Path to .pt checkpoint",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument(
        "--simplify", action="store_true", help="Simplify ONNX model"
    )
    parser.add_argument("--output", type=Path, default=None, help="Output path")
    args = parser.parse_args()

    # 加载模型
    from ultralytics import YOLO

    model = YOLO(str(args.weights))

    # 设置导出模式
    model.model.model[-1].export = True  # Pose head
    model.model.model[-1].format = "onnx"

    # 导出
    output = args.output or args.weights.with_suffix(".onnx")
    model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
    )

    print(f"Exported to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
