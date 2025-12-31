"""ONNX 模型量化 + 验证脚本

支持多种量化方式:
1. FP16 - 半精度浮点 (推荐, 兼容性最好)
2. 动态INT8 - 仅量化MatMul层 (跳过Conv)
3. 原始FP32 - 不量化
"""

import os
import onnx
import onnxruntime as ort
import numpy as np
import cv2

# ================== 配置 ==================
INPUT_MODEL = "pose_simplified.onnx"   # 原始模型
OUTPUT_FP16 = "pose_fp16.onnx"         # FP16模型
OUTPUT_INT8 = "pose_int8.onnx"         # INT8模型 (仅MatMul)
TEST_IMAGE = "test.png"                # 测试图像
IMGSZ = 416


def quantize_fp16():
    """FP16量化 - 兼容性最好"""
    from onnxconverter_common import float16

    print(f"\n🔄 FP16量化: {INPUT_MODEL} -> {OUTPUT_FP16}")

    model = onnx.load(INPUT_MODEL)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, OUTPUT_FP16)

    in_size = os.path.getsize(INPUT_MODEL) / 1024 / 1024
    out_size = os.path.getsize(OUTPUT_FP16) / 1024 / 1024
    print(f"✅ 完成! {in_size:.2f}MB -> {out_size:.2f}MB (压缩 {in_size/out_size:.1f}x)")
    return OUTPUT_FP16


def quantize_int8_safe():
    """INT8动态量化 - 仅量化MatMul层,跳过Conv"""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"\n🔄 INT8量化(仅MatMul): {INPUT_MODEL} -> {OUTPUT_INT8}")

    quantize_dynamic(
        model_input=INPUT_MODEL,
        model_output=OUTPUT_INT8,
        weight_type=QuantType.QUInt8,  # 使用QUInt8兼容性更好
        op_types_to_quantize=['MatMul', 'Gemm'],  # 只量化全连接层,跳过Conv
    )

    in_size = os.path.getsize(INPUT_MODEL) / 1024 / 1024
    out_size = os.path.getsize(OUTPUT_INT8) / 1024 / 1024
    print(f"✅ 完成! {in_size:.2f}MB -> {out_size:.2f}MB (压缩 {in_size/out_size:.1f}x)")
    return OUTPUT_INT8


def validate(model_path):
    """验证模型推理"""
    print(f"\n📦 加载模型: {model_path}")

    try:
        session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False

    # 获取输入信息
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    print(f"   输入: {input_name} {input_info.shape}")

    # 获取输出信息
    for out in session.get_outputs():
        print(f"   输出: {out.name} {out.shape}")

    # 加载测试图像
    if os.path.exists(TEST_IMAGE):
        print(f"🖼️  测试图像: {TEST_IMAGE}")
        img = cv2.imread(TEST_IMAGE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMGSZ, IMGSZ))
    else:
        print(f"🎲 使用随机输入 (未找到 {TEST_IMAGE})")
        img = np.random.randint(0, 255, (IMGSZ, IMGSZ, 3), dtype=np.uint8)

    # 预处理
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[None]  # (1, 3, H, W)

    # 推理
    try:
        outputs = session.run(None, {input_name: img})
        print(f"✅ 推理成功!")
        print(f"   输出形状: {[o.shape for o in outputs]}")
        return True
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        return False


def benchmark(model_path, runs=50):
    """性能测试"""
    import time

    try:
        session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
    except Exception as e:
        print(f"⚠️  {os.path.basename(model_path)}: 无法加载 - {e}")
        return None

    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, IMGSZ, IMGSZ).astype(np.float32)

    # 预热
    for _ in range(5):
        session.run(None, {input_name: dummy})

    # 测速
    t0 = time.time()
    for _ in range(runs):
        session.run(None, {input_name: dummy})
    ms = (time.time() - t0) / runs * 1000

    name = os.path.basename(model_path)
    size = os.path.getsize(model_path) / 1024 / 1024
    print(f"⏱️  {name}: {ms:.2f}ms ({1000/ms:.1f} FPS) | {size:.2f}MB")
    return ms


def main():
    if not os.path.exists(INPUT_MODEL):
        print(f"❌ 找不到输入模型: {INPUT_MODEL}")
        return

    print("=" * 50)
    print("ONNX 模型量化工具")
    print("=" * 50)

    # 验证原始模型
    print("\n[1/4] 验证原始模型")
    if not validate(INPUT_MODEL):
        print("原始模型无法运行,请检查")
        return

    # FP16量化 (推荐)
    print("\n[2/4] FP16量化")
    try:
        fp16_model = quantize_fp16()
        fp16_ok = validate(fp16_model)
    except ImportError:
        print("⚠️  需要安装 onnxconverter-common: pip install onnxconverter-common")
        fp16_ok = False
    except Exception as e:
        print(f"⚠️  FP16量化失败: {e}")
        fp16_ok = False

    # INT8量化 (仅MatMul)
    print("\n[3/4] INT8量化(安全模式)")
    try:
        int8_model = quantize_int8_safe()
        int8_ok = validate(int8_model)
    except Exception as e:
        print(f"⚠️  INT8量化失败: {e}")
        int8_ok = False

    # 性能对比
    print("\n[4/4] 性能对比")
    print("-" * 50)
    benchmark(INPUT_MODEL)
    if fp16_ok:
        benchmark(OUTPUT_FP16)
    if int8_ok:
        benchmark(OUTPUT_INT8)

    # 总结
    print("\n" + "=" * 50)
    print("推荐使用:")
    if fp16_ok:
        print(f"  ✅ {OUTPUT_FP16} (FP16, 兼容性好, 约50%压缩)")
    if int8_ok:
        print(f"  ✅ {OUTPUT_INT8} (INT8部分量化)")
    print(f"  ✅ {INPUT_MODEL} (原始FP32, 最高精度)")


if __name__ == "__main__":
    main()
