"""PT -> ONNX 导出脚本 (支持自定义 Pose head)"""
import argparse
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.modules.head import Pose


def export_onnx(weights, imgsz=640, simplify=True, opset=12):
    """导出 ONNX 模型"""
    model = YOLO(weights)

    # 为自定义 Pose head 设置 export 标志
    for m in model.model.modules():
        if isinstance(m, Pose):
            m.export = True
            m.format = 'onnx'
            print(f"已设置 Pose head export=True")

    # 导出
    model.export(
        format='onnx',
        imgsz=imgsz,
        simplify=simplify,
        opset=opset,
    )

    onnx_path = Path(weights).with_suffix('.onnx')
    print(f"导出完成: {onnx_path}")
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description='PT -> ONNX 导出')
    parser.add_argument('--weights', type=str, required=True, help='模型路径')
    parser.add_argument('--imgsz', type=int, default=640, help='输入尺寸')
    parser.add_argument('--simplify', action='store_true', default=True, help='简化模型')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset版本')
    args = parser.parse_args()

    export_onnx(args.weights, args.imgsz, args.simplify, args.opset)


if __name__ == '__main__':
    main()
