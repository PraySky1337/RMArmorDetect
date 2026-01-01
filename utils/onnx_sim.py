"""Simplify ONNX model using onnxsim for OpenVINO deployment."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Simplify ONNX model")
    parser.add_argument("input", type=Path, help="Input ONNX model path")
    parser.add_argument(
        "--output", "-o", type=Path, default=None, help="Output path (default: input_sim.onnx)"
    )
    parser.add_argument(
        "--check", action="store_true", help="Verify simplified model"
    )
    args = parser.parse_args()

    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print("Please install: pip install onnx onnxsim", file=sys.stderr)
        return 1

    # 加载模型
    print(f"Loading {args.input}...")
    model = onnx.load(str(args.input))

    # 简化
    print("Simplifying...")
    model_sim, check = simplify(model)

    if not check:
        print("Warning: Simplified model check failed!", file=sys.stderr)

    # 保存
    output = args.output or args.input.with_stem(args.input.stem + "_sim")
    onnx.save(model_sim, str(output))
    print(f"Saved to {output}")

    # 验证
    if args.check:
        print("Verifying model...")
        onnx.checker.check_model(model_sim)
        print("Model check passed!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
