# #!/usr/bin/env python3
# """Utility to export a pose model checkpoint (PyTorch) to ONNX."""

# # usage python utils/export.py --checkpoint path/to/weights.pth --model models.pose_model.Model --model-kwargs '{"num_joints":4, "num_classes":52}' --input-shape 1 3 640 640 --out pose.onnx
# # or using an Ultralytics YOLO pose checkpoint directly:
# # python utils/export.py --checkpoint best.pt --yolo --input-shape 1 3 640 640 --out pose.onnx

# from __future__ import annotations

# import argparse
# import importlib
# import json
# import sys
# from pathlib import Path
# from typing import Any, Dict, Iterable, Tuple

# import torch


# def import_from_string(target: str) -> Any:
#     """Import a class or factory given 'package.module:object' or 'package.module.object'."""
#     if ":" in target:
#         module_name, attr = target.split(":")
#     else:
#         module_name, attr = target.rsplit(".", 1)
#     module = importlib.import_module(module_name)
#     return getattr(module, attr)


# def parse_model_kwargs(raw: str | None) -> Dict[str, Any]:
#     if not raw:
#         return {}
#     try:
#         return json.loads(raw)
#     except json.JSONDecodeError as exc:  # pragma: no cover - CLI validation
#         raise SystemExit(f"--model-kwargs must be JSON (e.g., '{{\"num_joints\":17}}'): {exc}")


# def load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
#     payload = torch.load(path, map_location="cpu")
#     if isinstance(payload, dict):
#         for key in ("state_dict", "model", "ema"):
#             if key in payload and isinstance(payload[key], dict):
#                 payload = payload[key]
#                 break
#     if not isinstance(payload, dict):
#         raise TypeError(f"Checkpoint at {path} is not a state_dict-like mapping")

#     cleaned = {}
#     for k, v in payload.items():
#         nk = k.replace("module.", "", 1) if k.startswith("module.") else k  # strip DataParallel prefix
#         cleaned[nk] = v
#     return cleaned


# def build_model(model_target: str, checkpoint: Path, model_kwargs: Dict[str, Any], strict: bool) -> torch.nn.Module:
#     ctor_or_cls = import_from_string(model_target)
#     model = ctor_or_cls(**model_kwargs)
#     state = load_state_dict(checkpoint)
#     missing, unexpected = model.load_state_dict(state, strict=strict)
#     if missing:
#         print(f"[warn] Missing keys: {missing}", file=sys.stderr)
#     if unexpected:
#         print(f"[warn] Unexpected keys: {unexpected}", file=sys.stderr)
#     model.eval()
#     return model


# def infer_output_names(sample_output: Any, user_names: Iterable[str] | None) -> Tuple[Any, Tuple[str, ...]]:
#     if user_names:
#         return sample_output, tuple(user_names)

#     if isinstance(sample_output, (list, tuple)):
#         names = tuple(f"output{i}" for i in range(len(sample_output)))
#     else:
#         names = ("output",)
#     return sample_output, names


# def export_onnx(
#     model: torch.nn.Module,
#     dummy_input: torch.Tensor,
#     output_path: Path,
#     opset: int,
#     dynamic_batch: bool,
#     input_names: Tuple[str, ...],
#     output_names: Tuple[str, ...],
# ) -> None:
#     dynamic_axes = None
#     if dynamic_batch:
#         dynamic_axes = {name: {0: "batch"} for name in input_names}
#         for name in output_names:
#             dynamic_axes[name] = {0: "batch"}

#     torch.onnx.export(
#         model,
#         dummy_input,
#         output_path,
#         export_params=True,
#         opset_version=opset,
#         do_constant_folding=True,
#         input_names=list(input_names),
#         output_names=list(output_names),
#         dynamic_axes=dynamic_axes,
#     )


# def parse_args(argv: Iterable[str]) -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Export a pose model checkpoint (PyTorch) to ONNX."
#     )
#     parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pth/.pt checkpoint.")
#     parser.add_argument(
#         "--model",
#         type=str,
#         required=False,
#         help="Dotted path to model class/factory, e.g. 'models.pose_model.Model'. Not required when --yolo is set.",
#     )
#     parser.add_argument(
#         "--model-kwargs",
#         type=str,
#         default=None,
#         help='JSON string of kwargs passed to model ctor, e.g. \'{"num_joints":17}\'',
#     )
#     parser.add_argument(
#         "--num-joints",
#         type=int,
#         default=4,
#         help="Number of keypoints (default: 4). Used if --model-kwargs does not already set it.",
#     )
#     parser.add_argument(
#         "--num-classes",
#         type=int,
#         default=52,
#         help="Number of classes (default: 52). Used if --model-kwargs does not already set it.",
#     )
#     parser.add_argument(
#         "--yolo",
#         action="store_true",
#         help="Use Ultralytics YOLO export instead of importing a model class.",
#     )
#     parser.add_argument(
#         "--input-shape",
#         type=int,
#         nargs="+",
#         default=[1, 3, 640, 640],
#         help="Input shape as N C H W (default: 1 3 640 640).",
#     )
#     parser.add_argument(
#         "--device", type=str, default="cpu", help="Device for export dummy input (default: cpu)."
#     )
#     parser.add_argument(
#         "--opset", type=int, default=13, help="ONNX opset version (default: 13)."
#     )
#     parser.add_argument(
#         "--input-names",
#         type=str,
#         nargs="+",
#         default=["input"],
#         help="ONNX input tensor names (default: input).",
#     )
#     parser.add_argument(
#         "--output-names",
#         type=str,
#         nargs="*",
#         default=None,
#         help="ONNX output tensor names (default: auto based on model output count).",
#     )
#     parser.add_argument(
#         "--strict",
#         action="store_true",
#         help="Use strict=True when loading state_dict (default: False).",
#     )
#     parser.add_argument(
#         "--out",
#         type=Path,
#         default=Path("model.onnx"),
#         help="Output ONNX path (default: model.onnx).",
#     )
#     return parser.parse_args(list(argv))


# def main(argv: Iterable[str]) -> int:
#     args = parse_args(argv)
#     if len(args.input_shape) != 4:
#         print("Input shape must be four integers: N C H W", file=sys.stderr)
#         return 1
#     if not args.yolo and not args.model:
#         print("Argument --model is required unless --yolo is set.", file=sys.stderr)
#         return 1

#     if args.yolo:
#         try:
#             from ultralytics import YOLO
#         except Exception as e:  # pragma: no cover - import guard
#             print(f"Failed to import Ultralytics YOLO: {e}", file=sys.stderr)
#             return 1

#         if not args.checkpoint.exists():
#             print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
#             return 1

#         imgsz = tuple(args.input_shape[2:])
#         exported = Path(
#             YOLO(str(args.checkpoint)).export(
#                 format="onnx",
#                 imgsz=imgsz if len(imgsz) == 2 else imgsz[-1],
#                 opset=args.opset,
#                 device=args.device,
#             )
#         )
#         out_path = args.out
#         if exported != out_path:
#             out_path.parent.mkdir(parents=True, exist_ok=True)
#             out_path.write_bytes(exported.read_bytes())
#             print(f"Exported via Ultralytics YOLO to {exported} -> copied to {out_path}")
#         else:
#             print(f"Exported via Ultralytics YOLO to {exported}")
#         return 0

#     model_kwargs = parse_model_kwargs(args.model_kwargs)
#     # Inject common defaults if the user did not supply them explicitly.
#     model_kwargs.setdefault("num_joints", args.num_joints)
#     model_kwargs.setdefault("num_classes", args.num_classes)
#     model = build_model(args.model, args.checkpoint, model_kwargs, strict=args.strict)
#     model.to(args.device)

#     dummy = torch.randn(*args.input_shape, device=args.device)
#     with torch.no_grad():
#         sample_out = model(dummy)
#     sample_out, output_names = infer_output_names(sample_out, args.output_names)

#     export_onnx(
#         model=model,
#         dummy_input=dummy,
#         output_path=args.out,
#         opset=args.opset,
#         dynamic_batch=False,
#         input_names=tuple(args.input_names),
#         output_names=output_names,
#     )
#     print(f"Exported ONNX model to {args.out}")
#     return 0


# if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
#     raise SystemExit(main(sys.argv[1:]))


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
