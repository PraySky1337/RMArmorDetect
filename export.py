from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils import YAML

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"Disable {help_text.lower()}.")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO26 pose .pt to ONNX with int8/fp16 preference.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/pose/train/weights/best.pt"),
        help="Path to .pt pose model.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data.yaml"),
        help="Dataset yaml for INT8 calibration (uses train split by default).",
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=None,
        help="Optional directory of calibration images (overrides --data).",
    )
    parser.add_argument("--calib-max", type=int, default=300, help="Max calibration images (recommended >=300).")
    parser.add_argument("--calib-seed", type=int, default=0, help="Shuffle seed for calibration sampling.")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size (matches train.py default).")
    parser.add_argument("--batch", type=int, default=1, help="Export batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for export, e.g. '0' or 'cpu'. Defaults to CUDA if available.",
    )
    _add_bool_arg(parser, "dynamic", False, "Enable dynamic axes.")
    _add_bool_arg(parser, "simplify", True, "Enable ONNX simplify.")
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset (default: auto).")
    _add_bool_arg(parser, "end2end", True, "Keep end2end export (recommended for YOLO26 pose).")
    parser.add_argument(
        "--precision",
        choices=("auto", "int8", "fp16", "fp32"),
        default="auto",
        help="Preferred precision: auto=fp16->int8->fp32.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Final ONNX output path (defaults to model stem with suffix).",
    )
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate ONNX files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if exists.")
    return parser.parse_args()


def _default_device(device: str | None) -> str:
    if device:
        return device
    return "0" if torch.cuda.is_available() else "cpu"


def _collect_images(root: Path, max_count: int, seed: int) -> list[Path]:
    if not root or not root.exists():
        return []
    images = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        return []
    rnd = random.Random(seed)
    rnd.shuffle(images)
    return images[:max_count] if max_count > 0 else images


def _resolve_calib_dir(data_yaml: Path, calib_dir: Path | None) -> Path | None:
    if calib_dir is not None:
        return calib_dir
    if not data_yaml.exists():
        return None
    data = YAML.load(data_yaml)
    base = data_yaml.parent
    root = Path(data.get("path", "")) if isinstance(data, dict) else Path("")
    if root and not root.is_absolute():
        root = (base / root).resolve()
    train = data.get("train") if isinstance(data, dict) else None
    if isinstance(train, (list, tuple)):
        train = train[0] if train else None
    if not train:
        return None
    train_path = Path(train)
    if not train_path.is_absolute():
        train_path = (root / train_path) if root else (base / train_path)
    return train_path


def _export_fp32(model: YOLO, export_kwargs: dict) -> Path:
    fp32_path = Path(model.export(**export_kwargs, half=False, int8=False))
    if not fp32_path.exists():
        raise FileNotFoundError(f"FP32 export failed: {fp32_path}")
    return fp32_path


def _convert_fp16(fp32_path: Path, fp16_path: Path) -> bool:
    try:
        import onnx
        from onnxruntime.transformers import float16

        model_onnx = onnx.load(fp32_path)
        model_fp16 = float16.convert_float_to_float16(model_onnx, keep_io_types=True)
        onnx.save(model_fp16, fp16_path)
        return fp16_path.exists()
    except Exception as exc:
        print(f"[fp16] conversion failed: {exc}")
        return False


def _export_fp16_from_model(model: YOLO, export_kwargs: dict, fp32_path: Path, fp16_path: Path) -> bool:
    backup = fp32_path.with_suffix(".fp32.onnx")
    try:
        shutil.copy2(fp32_path, backup)
        exported = Path(model.export(**export_kwargs, half=True, int8=False))
        if exported.exists():
            exported.replace(fp16_path)
            return True
        return False
    except Exception as exc:
        print(f"[fp16] re-export failed: {exc}")
        return False
    finally:
        if backup.exists() and not fp32_path.exists():
            backup.replace(fp32_path)


def _copy_onnx_metadata(src: Path, dst: Path) -> None:
    try:
        import onnx

        src_model = onnx.load(src)
        dst_model = onnx.load(dst)
        if src_model.metadata_props and not dst_model.metadata_props:
            dst_model.metadata_props.extend(src_model.metadata_props)
            onnx.save(dst_model, dst)
    except Exception:
        pass


def _quantize_int8(fp32_path: Path, int8_path: Path, imgs: list[Path], imgsz: int, stride: int) -> bool:
    if not imgs:
        print("[int8] no calibration images found, skip.")
        return False
    try:
        import cv2
        import numpy as np
        import onnx
        from onnxruntime.quantization import (  # type: ignore
            CalibrationDataReader,
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
        from ultralytics.data.augment import LetterBox
    except Exception as exc:
        print(f"[int8] missing dependencies: {exc}")
        return False

    input_name = onnx.load(fp32_path).graph.input[0].name
    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=stride)

    class _ImageReader(CalibrationDataReader):
        def __init__(self, paths: list[Path]):
            self.paths = paths
            self._iter = iter(self.paths)

        def get_next(self):
            for p in self._iter:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = letterbox(image=img)
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))[None]
                return {input_name: img}
            return None

    reader = _ImageReader(imgs)
    try:
        quantize_static(
            fp32_path,
            int8_path,
            reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            per_channel=True,
        )
        if int8_path.exists():
            _copy_onnx_metadata(fp32_path, int8_path)
        return int8_path.exists()
    except Exception as exc:
        print(f"[int8] quantization failed: {exc}")
        return False


def _finalize_output(final_path: Path, output: Path | None, overwrite: bool) -> Path:
    if output is None:
        return final_path
    output = output.resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output} (use --overwrite to replace)")
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    final_path.replace(output)
    return output


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    device = _default_device(args.device)
    model = YOLO(str(args.model))
    stride = int(max(getattr(model.model, "stride", [32])))

    export_kwargs = dict(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
        end2end=args.end2end,
        nms=False,
    )

    fp32_path = _export_fp32(model, export_kwargs)
    cleanup: list[Path] = []
    final_path = fp32_path

    precision = args.precision
    if precision in ("auto", "fp16"):
        fp16_path = fp32_path.with_name(fp32_path.stem + "_fp16.onnx")
        if _convert_fp16(fp32_path, fp16_path):
            print(f"[fp16] export success: {fp16_path}")
            final_path = fp16_path
            cleanup.append(fp32_path)
        elif device != "cpu" and _export_fp16_from_model(model, export_kwargs, fp32_path, fp16_path):
            print(f"[fp16] export success: {fp16_path}")
            final_path = fp16_path
            cleanup.append(fp32_path)
        elif precision == "fp16":
            raise RuntimeError("FP16 export requested but failed.")

    if final_path == fp32_path and precision in ("auto", "int8"):
        calib_root = _resolve_calib_dir(args.data, args.calib_dir)
        imgs = _collect_images(calib_root, args.calib_max, args.calib_seed) if calib_root else []
        int8_path = fp32_path.with_name(fp32_path.stem + "_int8.onnx")
        if _quantize_int8(fp32_path, int8_path, imgs, args.imgsz, stride):
            print(f"[int8] export success: {int8_path}")
            final_path = int8_path
            cleanup.append(fp32_path)
        elif precision == "int8":
            raise RuntimeError("INT8 export requested but failed.")

    final_path = _finalize_output(final_path, args.output, args.overwrite)

    if cleanup and not args.keep_intermediate:
        for p in cleanup:
            if p.exists() and p != final_path:
                p.unlink()

    print(f"[done] final model: {final_path}")


if __name__ == "__main__":
    main()
