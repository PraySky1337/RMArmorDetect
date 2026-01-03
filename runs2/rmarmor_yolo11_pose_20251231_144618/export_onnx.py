#!/usr/bin/env python3
"""导出 best.pt 为 ONNX 格式"""

from ultralytics import YOLO

def main():
    # 加载模型
    model = YOLO("weights/best.pt")

    # 导出为 ONNX 格式
    model.export(
        format="onnx",
        opset=12,
        simplify=True,
        dynamic=False,
        imgsz=640,
    )

    print("导出完成！ONNX 文件保存在 weights/best.onnx")

if __name__ == "__main__":
    main()
