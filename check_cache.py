#!/usr/bin/env python3
"""诊断脚本：检查 cache="ram" 是否生效"""

import sys
import time
import psutil
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils import DEFAULT_CFG
from armor_detect.data.armor_dataset import ArmorDataset

def check_cache():
    """检查缓存是否生效"""

    # 1. 加载配置
    print("=" * 60)
    print("1. 加载配置")
    print("=" * 60)

    cfg = get_cfg(DEFAULT_CFG)
    cfg.merge(vars(get_cfg('train_config.yaml')))

    print(f"cache 配置: {cfg.cache}")
    print(f"imgsz: {cfg.imgsz}")
    print(f"batch: {cfg.batch}")
    print(f"workers: {cfg.workers}")

    # 2. 获取数据集路径
    data_file = Path('armor_detect/configs/armor_yolo11.yaml')
    from ultralytics.data.utils import check_det_dataset
    data_dict = check_det_dataset(str(data_file))

    train_path = data_dict['train']
    print(f"\n训练集路径: {train_path}")

    # 3. 记录初始内存
    print("\n" + "=" * 60)
    print("2. 创建 Dataset 前的内存状态")
    print("=" * 60)
    mem_before = psutil.virtual_memory()
    print(f"可用内存: {mem_before.available / (1<<30):.2f}GB")
    print(f"已用内存: {mem_before.used / (1<<30):.2f}GB")
    print(f"内存使用率: {mem_before.percent}%")

    # 4. 创建 Dataset（这里会触发缓存）
    print("\n" + "=" * 60)
    print("3. 创建 Dataset (应该会缓存图片)")
    print("=" * 60)
    print("等待 Dataset 初始化...")

    start_time = time.time()

    dataset = ArmorDataset(
        img_path=train_path,
        imgsz=cfg.imgsz,
        batch_size=cfg.batch,
        augment=True,
        hyp=cfg,
        rect=False,
        cache=cfg.cache,  # 显式传入 cache="ram"
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="train: ",
        task='pose',
        classes=None,
        data=data_dict,
        fraction=cfg.train_ratio,
    )

    init_time = time.time() - start_time

    # 5. 检查缓存后的内存
    print("\n" + "=" * 60)
    print("4. Dataset 创建后的内存状态")
    print("=" * 60)
    mem_after = psutil.virtual_memory()
    print(f"可用内存: {mem_after.available / (1<<30):.2f}GB")
    print(f"已用内存: {mem_after.used / (1<<30):.2f}GB")
    print(f"内存使用率: {mem_after.percent}%")

    mem_used = (mem_after.used - mem_before.used) / (1 << 30)
    print(f"\n新增内存占用: {mem_used:.2f}GB")
    print(f"初始化耗时: {init_time:.2f}秒")

    # 6. 检查 dataset 内部状态
    print("\n" + "=" * 60)
    print("5. Dataset 内部状态检查")
    print("=" * 60)
    print(f"dataset.cache: {dataset.cache}")
    print(f"dataset.ni (图片总数): {dataset.ni}")

    # 检查 self.ims 是否被填充
    cached_count = sum(1 for im in dataset.ims if im is not None)
    print(f"已缓存图片数: {cached_count} / {dataset.ni}")

    if cached_count == dataset.ni:
        print("✅ 所有图片已缓存到内存")
    elif cached_count == 0:
        print("❌ 没有图片被缓存！cache='ram' 未生效")
    else:
        print(f"⚠️ 部分图片被缓存 ({cached_count}/{dataset.ni})")

    # 7. 测试数据加载速度
    print("\n" + "=" * 60)
    print("6. 测试数据加载速度")
    print("=" * 60)

    print("测试加载 100 张图片...")
    start_time = time.time()
    for i in range(min(100, len(dataset))):
        _ = dataset[i]
    load_time = time.time() - start_time

    print(f"加载 100 张图片耗时: {load_time:.3f}秒")
    print(f"平均每张: {load_time / 100 * 1000:.2f}ms")

    if load_time < 1.0:  # 如果100张图片小于1秒，说明是从内存读取
        print("✅ 数据加载非常快，应该是从内存读取")
    else:
        print("⚠️ 数据加载较慢，可能没有使用缓存")

    # 8. 总结
    print("\n" + "=" * 60)
    print("诊断总结")
    print("=" * 60)

    if cached_count == dataset.ni and mem_used > 5.0:
        print("✅ cache='ram' 正常工作")
        print(f"   - 所有图片已加载到内存 ({mem_used:.2f}GB)")
        print(f"   - 数据加载速度快 ({load_time / 100 * 1000:.2f}ms/张)")
    else:
        print("❌ cache='ram' 可能未生效")
        if cached_count == 0:
            print("   - 原因: 没有图片被缓存")
        if mem_used < 5.0:
            print("   - 原因: 内存占用增长太少")
        print("\n可能的原因:")
        print("1. check_cache_ram() 返回 False (内存不足)")
        print("2. cache 参数未正确传递到 Dataset")
        print("3. 缓存逻辑被跳过")

if __name__ == "__main__":
    try:
        check_cache()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
