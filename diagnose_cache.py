"""诊断脚本：检查cache="ram"是否生效

运行此脚本可以诊断：
1. cache参数是否正确传递
2. check_cache_ram是否通过
3. 数据是否真的被缓存到RAM
4. DataLoader worker数量
"""

import os
import sys
import yaml
from pathlib import Path

# 添加项目路径
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ultralytics.cfg import IterableSimpleNamespace
from ultralytics.data.build import build_yolo_dataset
from ultralytics.utils import LOGGER

def main():
    print("="*80)
    print("RMArmorDetect 缓存诊断工具")
    print("="*80)

    # 1. 检查配置
    print("\n[1/5] 检查train_config.yaml配置...")
    train_cfg_path = ROOT / "train_config.yaml"
    with open(train_cfg_path) as f:
        cfg = yaml.safe_load(f)

    cache_config = cfg.get("cache", None)
    workers_config = cfg.get("workers", 8)
    batch_config = cfg.get("batch", 256)

    print(f"  ✓ cache配置: {cache_config}")
    print(f"  ✓ workers配置: {workers_config}")
    print(f"  ✓ batch配置: {batch_config}")

    if cache_config != "ram":
        print(f"  ⚠️  警告: cache应该是'ram'，但当前是'{cache_config}'")

    # 2. 检查系统资源
    print("\n[2/5] 检查系统资源...")
    import psutil
    mem = psutil.virtual_memory()
    cpu_count = os.cpu_count()

    print(f"  ✓ CPU核心数: {cpu_count}")
    print(f"  ✓ 总内存: {mem.total / 1024**3:.1f} GB")
    print(f"  ✓ 可用内存: {mem.available / 1024**3:.1f} GB")
    print(f"  ✓ 实际workers数: min({cpu_count}, {workers_config}) = {min(cpu_count, workers_config)}")

    if min(cpu_count, workers_config) < 16:
        print(f"  ⚠️  警告: workers数量={min(cpu_count, workers_config)}可能不够，建议至少16")

    # 3. 检查数据集
    print("\n[3/5] 检查数据集...")
    img_dir = ROOT / "datasets" / "images"
    if not img_dir.exists():
        print(f"  ✗ 错误: {img_dir}不存在")
        return

    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    total_images = len(img_files)
    print(f"  ✓ 图片数量: {total_images}")

    # 估算缓存需求
    if total_images > 0:
        import cv2
        import random
        sample_imgs = random.sample(img_files, min(30, total_images))
        total_bytes = 0
        for img_path in sample_imgs:
            im = cv2.imread(str(img_path))
            if im is not None:
                ratio = 416 / max(im.shape[0], im.shape[1])
                total_bytes += im.nbytes * ratio**2

        avg_bytes = total_bytes / len(sample_imgs)
        estimated_ram = avg_bytes * total_images * 1.5 / 1024**3  # 1.5 for safety margin
        print(f"  ✓ 预估缓存需求: {estimated_ram:.1f} GB")

        if estimated_ram > mem.available / 1024**3:
            print(f"  ✗ 错误: 需要{estimated_ram:.1f}GB，但只有{mem.available/1024**3:.1f}GB可用")
            print(f"     check_cache_ram将返回False，缓存不会生效！")
        else:
            print(f"  ✓ 内存充足，缓存应该能生效")

    # 4. 测试创建dataset
    print("\n[4/5] 测试创建Dataset...")
    try:
        # 加载data.yaml
        data_yaml = ROOT / "datasets" / "rmarmor-pose.yaml"
        if not data_yaml.exists():
            print(f"  ⚠️  {data_yaml}不存在，跳过dataset创建测试")
            return

        with open(data_yaml) as f:
            data = yaml.safe_load(f)

        # 创建配置
        args = IterableSimpleNamespace(
            imgsz=416,
            cache="ram",
            rect=False,
            single_cls=False,
            task="pose",
            classes=None,
            fraction=0.01,  # 只用1%的数据测试
        )

        train_path = data.get("train", "")
        if not train_path:
            print(f"  ⚠️  data.yaml中没有train字段")
            return

        print(f"  → 创建测试dataset（使用1%数据）...")
        dataset = build_yolo_dataset(
            args,
            train_path,
            batch=batch_config,
            data=data,
            mode="train",
            stride=32
        )

        print(f"  ✓ Dataset创建成功")
        print(f"  ✓ 图片数量: {dataset.ni}")
        print(f"  ✓ Cache设置: {dataset.cache}")

        # 检查是否真的缓存了
        if dataset.cache == "ram":
            cached_count = sum(1 for im in dataset.ims if im is not None)
            print(f"  ✓ 已缓存图片: {cached_count}/{dataset.ni}")

            if cached_count == 0:
                print(f"  ✗ 错误: cache='ram'但没有图片被缓存！")
                print(f"     可能原因: check_cache_ram()返回了False")
            elif cached_count < dataset.ni:
                print(f"  ⚠️  警告: 只缓存了{cached_count}/{dataset.ni}张图片")
            else:
                print(f"  ✓ 所有图片已成功缓存到RAM！")
        else:
            print(f"  ✗ 错误: dataset.cache={dataset.cache}，不是'ram'")
            print(f"     缓存可能在check_cache_ram()阶段被禁用")

    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()

    # 5. 总结
    print("\n[5/5] 诊断总结")
    print("="*80)
    print("请检查以上输出，特别关注：")
    print("1. [3/5] 中'内存充足，缓存应该能生效'应该显示")
    print("2. [4/5] 中'所有图片已成功缓存到RAM！'应该显示")
    print("")
    print("如果看到任何'✗ 错误'或'⚠️  警告'，请将完整输出发送给开发者")
    print("="*80)

if __name__ == "__main__":
    main()
