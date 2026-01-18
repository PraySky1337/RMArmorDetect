#!/usr/bin/env python3
"""
诊断脚本：分析训练后的模型置信度和分类问题
用法: python diagnose_confidence.py <weights_path>
"""

import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from armor_detect.models import ArmorPoseModel
from armor_detect.data import ArmorDataset
from ultralytics.data.build import build_dataloader


def diagnose_confidence(weights_path: str, data_config: str = "armor_data.yaml"):
    """诊断模型的置信度分布问题"""

    # 1. 加载模型
    print(f"[1/5] 加载模型: {weights_path}")
    model = ArmorPoseModel(weights_path)
    model.eval()

    # 获取pose head
    pose_head = None
    for m in model.model.modules():
        if hasattr(m, 'nc_num'):
            pose_head = m
            break

    if not pose_head:
        print("错误：未找到ArmorPoseHead")
        return

    print(f"  - nc_num: {pose_head.nc_num}")
    print(f"  - nc_color: {pose_head.nc_color}")
    print(f"  - nc_size: {pose_head.nc_size}")

    # 2. 检查分类头的bias值
    print("\n[2/5] 检查分类头bias值:")
    num_bias = pose_head.cv3[0][-1].bias.data.mean().item()
    color_bias = pose_head.cv_color[0][-1].bias.data.mean().item()
    size_bias = pose_head.cv_size[0][-1].bias.data.mean().item()

    print(f"  - Number branch bias: {num_bias:.4f} (sigmoid: {torch.sigmoid(torch.tensor(num_bias)):.4f})")
    print(f"  - Color branch bias:  {color_bias:.4f} (sigmoid: {torch.sigmoid(torch.tensor(color_bias)):.4f})")
    print(f"  - Size branch bias:   {size_bias:.4f} (sigmoid: {torch.sigmoid(torch.tensor(size_bias)):.4f})")

    # 诊断：bias是否仍然很负
    if num_bias < -2.0:
        print("  ⚠️  Number branch bias 仍然很负，可能导致conf偏低")
    if color_bias < -2.0:
        print("  ⚠️  Color branch bias 仍然很负")

    # 3. 分析验证集上的置信度分布
    print("\n[3/5] 分析验证集置信度分布...")
    device = next(model.parameters()).device

    # 创建验证dataloader
    val_args = {
        'data': data_config,
        'batch': 32,
        'imgsz': 416,
        'conf': 0.001,  # 设置极低阈值以获取所有预测
        'iou': 0.5,
        'device': device,
    }

    try:
        val_loader = build_dataloader(val_args, mode='val')
    except Exception as e:
        print(f"  无法创建dataloader: {e}")
        print("  跳过验证集分析")
        val_loader = None

    if val_loader:
        all_num_conf = []
        all_color_conf = []
        all_size_conf = []
        all_combined_conf = []

        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="  推理中"):
                # 预处理
                imgs = batch['img'].to(device)

                # 前向传播
                preds = model(imgs)

                # 解析预测
                if isinstance(preds, tuple):
                    preds = preds[0]

                bs, channels, na = preds.shape
                nc_num = pose_head.nc_num
                nc_color = pose_head.nc_color
                nc_size = pose_head.nc_size

                # 分割预测
                for b_idx in range(bs):
                    pred_data = preds[b_idx]
                    num_scores = pred_data[:nc_num]
                    color_scores = pred_data[nc_num:nc_num + nc_color]
                    size_scores = pred_data[nc_num + nc_color:nc_num + nc_color + nc_size]

                    # 获取最大置信度
                    num_conf = num_scores.max().item()
                    color_conf = color_scores.max().item()
                    size_conf = size_scores.max().item()
                    combined = num_conf * color_conf * size_conf

                    all_num_conf.append(num_conf)
                    all_color_conf.append(color_conf)
                    all_size_conf.append(size_conf)
                    all_combined_conf.append(combined)

        # 统计
        print("\n  置信度分布统计:")
        print(f"  {'分支':<12} {'均值':<10} {'中位数':<10} {'75%分位':<10} {'最大值':<10}")
        print(f"  {'-'*50}")
        print(f"  {'Number':<12} {np.mean(all_num_conf):<10.4f} {np.median(all_num_conf):<10.4f} {np.percentile(all_num_conf, 75):<10.4f} {np.max(all_num_conf):<10.4f}")
        print(f"  {'Color':<12} {np.mean(all_color_conf):<10.4f} {np.median(all_color_conf):<10.4f} {np.percentile(all_color_conf, 75):<10.4f} {np.max(all_color_conf):<10.4f}")
        print(f"  {'Size':<12} {np.mean(all_size_conf):<10.4f} {np.median(all_size_conf):<10.4f} {np.percentile(all_size_conf, 75):<10.4f} {np.max(all_size_conf):<10.4f}")
        print(f"  {'Combined':<12} {np.mean(all_combined_conf):<10.4f} {np.median(all_combined_conf):<10.4f} {np.percentile(all_combined_conf, 75):<10.4f} {np.max(all_combined_conf):<10.4f}")

        # 诊断
        print("\n  诊断:")
        if np.median(all_num_conf) < 0.3:
            print("  ⚠️  Number分支置信度中位数 < 0.3，可能存在conf偏低问题")
        if np.median(all_color_conf) < 0.3:
            print("  ⚠️  Color分支置信度中位数 < 0.3")
        if np.median(all_combined_conf) < 0.1:
            print("  ⚠️  组合置信度中位数 < 0.1，建议使用相加而非相乘")

    # 4. 检查损失权重配置
    print("\n[4/5] 损失权重配置:")
    print("  当前配置 (train_config.yaml 或 armor_yolo11.yaml):")
    print("  - pose (keypoint loss): 2.0")
    print("  - kobj (objectness): 1.5")
    print("  - cls (number): 1.5")
    print("  - color: 1.0")
    print("  - size: 0.8")
    print("\n  分析:")
    print("  - 关键点相关总权重: 2.0 + 1.5 = 3.5")
    print("  - 分类相关总权重: 1.5 + 1.0 + 0.8 = 3.3")
    print("  - 比例接近平衡，但color权重相对较低")

    # 5. 推荐解决方案
    print("\n[5/5] 推荐解决方案:")

    print("\n  【针对conf偏低】:")
    print("  方案A: 提高初始置信度")
    print("    - 修改 armor_detect/utils/__init__.py: INITIAL_CONF = 0.01 → 0.1")
    print("    - 效果: 初始bias从 -4.6 变为 -2.2")
    print("  方案B: 降低focal_gamma")
    print("    - 修改 train_config.yaml: focal_gamma: 2.0 → 1.0")
    print("    - 效果: 减少对简单样本的压抑")
    print("  方案C: 使用相加组合方式")
    print("    - 修改 armor_validator.py:109")
    print("    - 从: combined_conf = num_conf * color_conf * size_conf")
    print("    - 改为: combined_conf = (num_conf + color_conf + size_conf) / 3")

    print("\n  【针对过度关注灯条特征】:")
    print("  方案D: 增加HSV增强（迫使网络学习形状而非颜色）")
    print("    - 修改 train_config.yaml:")
    print("      hsv_h: 0.005 → 0.02 (色相旋转)")
    print("      hsv_s: 0.15 → 0.5 (饱和度变化)")
    print("    - 效果: 打破'红色=装甲板'的浅层关联")
    print("  方案E: 提高color loss权重")
    print("    - 修改 armor_yolo11.yaml: color: 1.0 → 2.0")
    print("    - 同时提高: bg_weight: 0.1 → 0.3")
    print("    - 效果: 强化颜色分类，增加负样本约束")
    print("  方案F: 增加Color分支容量")
    print("    - 修改 armor_pose_head.py:100")
    print("    - 从: c3_color = max(ch[0] // 2, ...)")
    print("    - 改为: c3_color = max(ch[0], ...) (与num相同)")

    print("\n  【推荐组合方案】:")
    print("  保守方案 (低风险): 方案A + 方案D")
    print("  激进方案 (高收益): 方案A + 方案C + 方案D + 方案E")

    print("\n" + "="*60)
    print("诊断完成！请根据上述分析选择合适的方案。")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python diagnose_confidence.py <weights_path> [data_config]")
        print("示例: python diagnose_confidence.py runs/armor/weights/best.pt armor_data.yaml")
        sys.exit(1)

    weights_path = sys.argv[1]
    data_config = sys.argv[2] if len(sys.argv) > 2 else "armor_data.yaml"

    diagnose_confidence(weights_path, data_config)
