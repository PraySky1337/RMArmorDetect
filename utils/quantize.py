# """ONNX 模型静态量化脚本"""

# from __future__ import annotations

# import os
# import cv2
# import numpy as np
# from pathlib import Path
# from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

# # ================== 配置 ==================
# INPUT_MODEL = "pose_simplified.onnx"      # 输入 ONNX 模型
# OUTPUT_MODEL = "pose_int8.onnx"           # 输出量化模型
# CALIBRATION_DIR = "../img_quantize"       # 校准图像目录
# IMGSZ = 416                               # 输入尺寸
# NUM_CALIBRATION = 100                     # 校准图像数量


# class ImageCalibrationDataReader(CalibrationDataReader):
#     """从图像目录读取校准数据"""

#     def __init__(self, calibration_dir: str, imgsz: int = 416, num_samples: int = 100):
#         self.imgsz = imgsz
#         self.idx = 0

#         # 获取图像列表
#         valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
#         image_dir = Path(calibration_dir)
#         self.image_files = [
#             f for f in image_dir.iterdir()
#             if f.suffix.lower() in valid_exts
#         ][:num_samples]

#         print(f"📁 找到 {len(self.image_files)} 张校准图像")

#     def preprocess(self, image_path: Path) -> np.ndarray:
#         """预处理图像为模型输入格式"""
#         img = cv2.imread(str(image_path))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # Resize with letterbox
#         h, w = img.shape[:2]
#         scale = min(self.imgsz / h, self.imgsz / w)
#         nh, nw = int(h * scale), int(w * scale)
#         img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

#         # Pad to square
#         pad_h = (self.imgsz - nh) // 2
#         pad_w = (self.imgsz - nw) // 2
#         img_padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
#         img_padded[pad_h:pad_h + nh, pad_w:pad_w + nw] = img

#         # Normalize and transpose
#         img_padded = img_padded.astype(np.float32) / 255.0
#         img_padded = img_padded.transpose(2, 0, 1)  # HWC -> CHW
#         img_padded = np.expand_dims(img_padded, axis=0)  # Add batch dim

#         return img_padded

#     def get_next(self) -> dict | None:
#         if self.idx >= len(self.image_files):
#             return None

#         image_path = self.image_files[self.idx]
#         self.idx += 1

#         try:
#             data = self.preprocess(image_path)
#             # 输入名称需要与 ONNX 模型匹配
#             return {"images": data}
#         except Exception as e:
#             print(f"⚠️ 跳过图像 {image_path}: {e}")
#             return self.get_next()

#     def rewind(self):
#         self.idx = 0


# def quantize_model(
#     input_model: str,
#     output_model: str,
#     calibration_dir: str,
#     imgsz: int = 416,
#     num_calibration: int = 100,
# ):
#     """执行静态量化"""

#     print(f"📦 输入模型: {input_model}")
#     print(f"📦 输出模型: {output_model}")
#     print(f"📐 输入尺寸: {imgsz}")

#     # 创建校准数据读取器
#     calibration_reader = ImageCalibrationDataReader(
#         calibration_dir=calibration_dir,
#         imgsz=imgsz,
#         num_samples=num_calibration,
#     )

#     # 执行静态量化
#     print("🔄 开始量化...")
#     quantize_static(
#         model_input=input_model,
#         model_output=output_model,
#         calibration_data_reader=calibration_reader,
#         quant_format=QuantFormat.QDQ,  # 量化格式
#         weight_type=QuantType.QInt8,   # 权重量化类型
#         activation_type=QuantType.QUInt8,  # 激活量化类型
#         per_channel=True,              # 逐通道量化（更精确）
#         reduce_range=False,
#     )

#     print(f"✅ 量化完成: {output_model}")

#     # 打印模型大小对比
#     input_size = os.path.getsize(input_model) / (1024 * 1024)
#     output_size = os.path.getsize(output_model) / (1024 * 1024)
#     print(f"📊 原始大小: {input_size:.2f} MB")
#     print(f"📊 量化大小: {output_size:.2f} MB")
#     print(f"📊 压缩比: {input_size / output_size:.2f}x")


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="ONNX 模型静态量化")
#     parser.add_argument("--input", type=str, default=INPUT_MODEL, help="输入 ONNX 模型路径")
#     parser.add_argument("--output", type=str, default=OUTPUT_MODEL, help="输出量化模型路径")
#     parser.add_argument("--calib-dir", type=str, default=CALIBRATION_DIR, help="校准图像目录")
#     parser.add_argument("--imgsz", type=int, default=IMGSZ, help="输入图像尺寸")
#     parser.add_argument("--num-calib", type=int, default=NUM_CALIBRATION, help="校准图像数量")
#     args = parser.parse_args()

#     quantize_model(
#         input_model=args.input,
#         output_model=args.output,
#         calibration_dir=args.calib_dir,
#         imgsz=args.imgsz,
#         num_calibration=args.num_calib,
#     )

from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
      model_input="pose_simplified.onnx",
      model_output="pose_int8.onnx",
      weight_type=QuantType.QInt8
  )
print("✅ 量化完成")