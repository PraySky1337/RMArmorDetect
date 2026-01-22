#!/bin/bash
# YOLO26 升级验证脚本 (不依赖 PyTorch)

echo "============================================================"
echo "YOLO26 升级验证"
echo "============================================================"
echo ""

# 1. 检查版本号
echo "1. 版本检查"
echo "------------------------------------------------------------"
grep "__version__" ultralytics/__init__.py | grep -o '8\.[0-9]\+\.[0-9]\+'
echo ""

# 2. 检查 YOLO26 配置文件
echo "2. YOLO26 配置文件"
echo "------------------------------------------------------------"
if [ -d "ultralytics/cfg/models/26" ]; then
    echo "✓ YOLO26 配置目录存在"
    ls -1 ultralytics/cfg/models/26/*.yaml | wc -l | xargs echo "  配置文件数量:"
    ls -1 ultralytics/cfg/models/26/*.yaml
else
    echo "✗ YOLO26 配置目录不存在"
fi
echo ""

# 3. 检查 MuSGD 优化器
echo "3. MuSGD 优化器支持"
echo "------------------------------------------------------------"
if [ -f "ultralytics/optim/muon.py" ]; then
    echo "✓ Muon 优化器模块存在"
    grep -c "class Muon" ultralytics/optim/muon.py | xargs echo "  Muon 类定义:"
else
    echo "✗ Muon 优化器模块不存在"
fi
echo ""

# 4. 检查损失函数
echo "4. 损失函数"
echo "------------------------------------------------------------"
losses=("Pose26" "RLELoss" "v8PoseLoss")
for loss in "${losses[@]}"; do
    if grep -q "class $loss" ultralytics/utils/loss.py 2>/dev/null; then
        echo "✓ $loss 类存在"
    else
        echo "✗ $loss 类不存在"
    fi
done
echo ""

# 5. 检查 Head 类
echo "5. Head 类"
echo "------------------------------------------------------------"
heads=("Pose26" "OBB26" "Segment26" "Pose")
for head in "${heads[@]}"; do
    if grep -q "class $head" ultralytics/nn/modules/head.py 2>/dev/null; then
        echo "✓ $head 类存在"
    else
        echo "✗ $head 类不存在"
    fi
done
echo ""

# 6. 检查默认配置
echo "6. 默认配置参数"
echo "------------------------------------------------------------"
params=("multi_scale: 0.0" "rle: 1.0" "angle: 1.0" "color: 1.0" "size: 1.0")
for param in "${params[@]}"; do
    if grep -q "$param" ultralytics/cfg/default.yaml; then
        echo "✓ $param"
    else
        echo "✗ $param"
    fi
done
echo ""

# 7. 检查装甲板检测特性
echo "7. 装甲板检测特性保留"
echo "------------------------------------------------------------"
features=("wing_omega" "wing_epsilon" "focal_gamma")
for feature in "${features[@]}"; do
    if grep -q "$feature" ultralytics/cfg/default.yaml; then
        echo "✓ $feature 参数保留"
    else
        echo "✗ $feature 参数缺失"
    fi
done

# 检查 color/size 处理
if grep -q "Sync color/size" ultralytics/data/augment.py 2>/dev/null; then
    echo "✓ color/size 同步逻辑保留"
else
    echo "⚠ color/size 同步逻辑可能需要检查"
fi
echo ""

# 8. 检查冲突标记
echo "8. 检查冲突标记"
echo "------------------------------------------------------------"
conflicts=$(grep -r "<<<<<<< HEAD" ultralytics/ 2>/dev/null | wc -l)
if [ "$conflicts" -eq 0 ]; then
    echo "✓ 没有发现冲突标记"
else
    echo "✗ 发现 $conflicts 处冲突标记"
    grep -r "<<<<<<< HEAD" ultralytics/ 2>/dev/null
fi
echo ""

# 9. Python 语法检查
echo "9. Python 语法检查"
echo "------------------------------------------------------------"
python_files=(
    "ultralytics/nn/modules/head.py"
    "ultralytics/nn/tasks.py"
    "ultralytics/utils/loss.py"
    "ultralytics/data/augment.py"
)
all_ok=true
for file in "${python_files[@]}"; do
    if python -m py_compile "$file" 2>/dev/null; then
        echo "✓ $file"
    else
        echo "✗ $file 有语法错误"
        all_ok=false
    fi
done
echo ""

echo "============================================================"
echo "验证完成"
echo "============================================================"
