#!/usr/bin/env python3
"""简单测试：验证 cache="ram" 是否生效"""

import time
import psutil
from pathlib import Path

# 运行前请确保激活训练环境
print("请在训练环境中运行此脚本（需要 torch 等依赖）")
print("=" * 70)

# 记录初始内存
mem_before = psutil.virtual_memory()
print(f"初始内存状态:")
print(f"  可用: {mem_before.available / (1<<30):.2f}GB")
print(f"  已用: {mem_before.used / (1<<30):.2f}GB")
print(f"  使用率: {mem_before.percent}%")
print()

# 导入会触发一些内存分配
from armor_detect.data.armor_dataset import ArmorDataset
from ultralytics.data.utils import check_det_dataset

# 加载数据集配置
data_file = Path('armor_detect/configs/armor_yolo11.yaml')
data_dict = check_det_dataset(str(data_file))
train_path = data_dict['train']

print(f"数据集路径: {train_path}")
print()
print("=" * 70)
print("开始创建 Dataset (cache='ram')...")
print("=" * 70)

start_time = time.time()

# 创建 Dataset，启用 RAM 缓存
dataset = ArmorDataset(
    img_path=train_path,
    imgsz=416,
    batch_size=256,
    augment=False,  # 关闭增强以简化测试
    hyp=None,
    rect=False,
    cache="ram",  # 关键：启用 RAM 缓存
    single_cls=False,
    stride=32,
    pad=0.0,
    prefix="[TEST] ",
    task='pose',
    classes=None,
    data=data_dict,
    fraction=1.0,
)

init_time = time.time() - start_time

print()
print("=" * 70)
print("Dataset 创建完成")
print("=" * 70)

# 检查缓存后的内存
mem_after = psutil.virtual_memory()
print(f"当前内存状态:")
print(f"  可用: {mem_after.available / (1<<30):.2f}GB")
print(f"  已用: {mem_after.used / (1<<30):.2f}GB")
print(f"  使用率: {mem_after.percent}%")

mem_increase = (mem_after.used - mem_before.used) / (1 << 30)
print(f"\n内存增长: {mem_increase:.2f}GB")
print(f"初始化耗时: {init_time:.2f}秒")

# 检查内部状态
print()
print("=" * 70)
print("Dataset 内部状态")
print("=" * 70)
print(f"dataset.cache: {dataset.cache}")
print(f"dataset.ni (总图片数): {dataset.ni}")

cached_count = sum(1 for im in dataset.ims if im is not None)
print(f"已缓存图片数: {cached_count} / {dataset.ni}")

if cached_count == dataset.ni:
    print("✅ 结果: 所有图片已缓存到内存")
elif cached_count == 0:
    print("❌ 结果: 没有图片被缓存，cache='ram' 未生效")
else:
    print(f"⚠️ 结果: 部分缓存 ({cached_count}/{dataset.ni})")

# 测试数据加载速度
print()
print("=" * 70)
print("测试数据加载速度")
print("=" * 70)

n_test = min(100, len(dataset))
print(f"测试加载 {n_test} 张图片...")

start_time = time.time()
for i in range(n_test):
    _ = dataset[i]
load_time = time.time() - start_time

print(f"加载 {n_test} 张图片耗时: {load_time:.3f}秒")
print(f"平均每张: {load_time / n_test * 1000:.2f}ms")

# 判断
print()
print("=" * 70)
print("诊断结论")
print("=" * 70)

if cached_count == dataset.ni and mem_increase > 5.0:
    print("✅ cache='ram' 工作正常")
    print(f"   - 图片已全部缓存 ({cached_count}/{dataset.ni})")
    print(f"   - 内存占用符合预期 ({mem_increase:.2f}GB)")
    print(f"   - 数据加载速度: {load_time / n_test * 1000:.2f}ms/张")
elif cached_count == 0:
    print("❌ cache='ram' 未生效")
    print("   可能原因:")
    print("   1. check_cache_ram() 判断内存不足返回 False")
    print("   2. cache 参数传递有问题")
    print("   3. 看上面的日志输出，查找 'RAM check' 或 'Cache mode' 关键词")
else:
    print("⚠️ cache='ram' 部分工作")
    print(f"   已缓存: {cached_count}/{dataset.ni}")
    if mem_increase < 5.0:
        print(f"   ⚠️ 内存增长偏低 ({mem_increase:.2f}GB)，可能有问题")

print()
print("请检查上面的日志输出，特别关注:")
print("  - 'Cache mode: ram (input: ram)'")
print("  - 'RAM check: need XXX GB, available XXX GB'")
print("  - '✅ Cached XXX/XXX images to RAM'")
print()
