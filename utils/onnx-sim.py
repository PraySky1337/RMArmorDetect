import onnx
from onnxsim import simplify

model = onnx.load("/home/ljy/Desktop/RMArmorDetect/utils/best_fp32.onnx")

model_simp, check = simplify(
    model,
)

assert check, "Simplified ONNX model could not be validated"

onnx.save(model_simp, "pose_simplified.onnx")
print("Saved to pose_simplified.onnx")
