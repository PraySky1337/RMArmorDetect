import onnx
import numpy as np
from collections import defaultdict

# 加载 ONNX 模型
model_path = "/home/rry/ultralytics/0526.onnx"
model = onnx.load(model_path)

print("=" * 60)
print("ONNX 模型分析 - 0526.onnx")
print("=" * 60)

# 基本信息
print(f"\n【基本信息】")
print(f"ONNX 版本: {model.opset_import[0].version if model.opset_import else 'N/A'}")
print(f"生产者: {model.producer_name if model.producer_name else 'N/A'}")
print(f"图节点数量: {len(model.graph.node)}")

# 输入输出信息
print(f"\n【输入】")
for inp in model.graph.input:
    shape = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in inp.type.tensor_type.shape.dim]
    print(f"  - {inp.name}: {shape}")

print(f"\n【输出】")
for out in model.graph.output:
    shape = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in out.type.tensor_type.shape.dim]
    print(f"  - {out.name}: {shape}")

# 统计参数
print(f"\n【参数统计】")
total_params = 0
param_details = []

for init in model.graph.initializer:
    param_count = np.prod(init.dims)
    total_params += param_count
    param_details.append((init.name, list(init.dims), param_count))

print(f"总参数量: {total_params:,}")
print(f"参数量 (M): {total_params / 1e6:.2f}M")

# 参数详情
print(f"\n【参数详情】")
for name, dims, count in sorted(param_details, key=lambda x: -x[2])[:20]:
    print(f"  {name}: {dims} = {count:,}")

# 按层类型分析
print(f"\n【算子类型统计】")
op_count = defaultdict(int)
for node in model.graph.node:
    op_count[node.op_type] += 1

for op_type, count in sorted(op_count.items(), key=lambda x: -x[1]):
    print(f"  {op_type}: {count}")

# 架构概览
print(f"\n【网络架构】")
print(f"{'序号':<6} {'算子类型':<20} {'输入数量':<10} {'输出数量':<10}")
print("-" * 50)
for i, node in enumerate(model.graph.node):
    print(f"{i:<6} {node.op_type:<20} {len(node.input):<10} {len(node.output):<10}")

# 尝试估算 FLOPs (简化版本)
print(f"\n【FLOPs 估算】")
flops = 0

for node in model.graph.node:
    op_type = node.op_type

    # 获取输入输出形状 (简化估算)
    for init in model.graph.initializer:
        if init.name in node.input:
            dims = list(init.dims)

            if op_type == "Conv":
                # Conv FLOPs: output_elements * kernel_elements * input_channels / output_channels
                if len(dims) >= 4:
                    # [out_channels, in_channels, kernel_h, kernel_w]
                    conv_flops = np.prod(dims)
                    flops += conv_flops
            elif op_type == "MatMul" or op_type == "Gemm":
                flops += np.prod(dims)
            elif op_type in ["Add", "Mul", "Div", "Sub"]:
                flops += np.prod(dims)

print(f"估算 FLOPs: {flops:,}")
print(f"估算 GFLOPs: {flops / 1e9:.2f}G")
