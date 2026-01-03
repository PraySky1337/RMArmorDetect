#!/usr/bin/env python3
"""
清洗数据集中无法匹配的图片或标签文件。

- 删除没有对应标签的图片
- 删除没有对应图片的标签

用法:
    python clean_dataset.py --images /path/to/images --labels /path/to/labels
    python clean_dataset.py --images /path/to/images --labels /path/to/labels --dry-run  # 只预览不删除
"""
import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}


def main():
    parser = argparse.ArgumentParser(description='清洗数据集中无法匹配的图片或标签文件')
    parser.add_argument('--images', required=True, help='图片目录')
    parser.add_argument('--labels', required=True, help='标签目录')
    parser.add_argument('--dry-run', action='store_true', help='只预览不实际删除')
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)

    if not images_dir.exists():
        print(f"错误: 图片目录不存在: {images_dir}")
        return 1
    if not labels_dir.exists():
        print(f"错误: 标签目录不存在: {labels_dir}")
        return 1

    # 获取所有图片和标签的文件名（不含扩展名）
    image_files = {f for f in images_dir.iterdir() if f.suffix in IMAGE_EXTENSIONS}
    label_files = {f for f in labels_dir.iterdir() if f.suffix == '.txt'}

    image_stems = {f.stem for f in image_files}
    label_stems = {f.stem for f in label_files}

    # 找出不匹配的文件
    orphan_images = [f for f in image_files if f.stem not in label_stems]
    orphan_labels = [f for f in label_files if f.stem not in image_stems]

    print(f"图片总数: {len(image_files)}")
    print(f"标签总数: {len(label_files)}")
    print(f"无标签的图片: {len(orphan_images)}")
    print(f"无图片的标签: {len(orphan_labels)}")
    print()

    if not orphan_images and not orphan_labels:
        print("数据集已经是干净的，无需清洗")
        return 0

    if args.dry_run:
        print("=== DRY RUN 模式 (不实际删除) ===")

    # 删除无标签的图片
    if orphan_images:
        print(f"\n待删除的图片 ({len(orphan_images)}):")
        for f in sorted(orphan_images)[:10]:
            print(f"  {f.name}")
        if len(orphan_images) > 10:
            print(f"  ... 还有 {len(orphan_images) - 10} 个")

        if not args.dry_run:
            for f in orphan_images:
                f.unlink()
            print(f"已删除 {len(orphan_images)} 个无标签的图片")

    # 删除无图片的标签
    if orphan_labels:
        print(f"\n待删除的标签 ({len(orphan_labels)}):")
        for f in sorted(orphan_labels)[:10]:
            print(f"  {f.name}")
        if len(orphan_labels) > 10:
            print(f"  ... 还有 {len(orphan_labels) - 10} 个")

        if not args.dry_run:
            for f in orphan_labels:
                f.unlink()
            print(f"已删除 {len(orphan_labels)} 个无图片的标签")

    if args.dry_run:
        print("\n使用不带 --dry-run 的命令来实际执行删除")

    return 0


if __name__ == '__main__':
    exit(main())
