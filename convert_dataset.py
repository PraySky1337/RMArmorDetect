#!/usr/bin/env python3
"""
将外部数据集格式转换为本项目格式。

源数据集: 52类combined格式 (如 BGs, R1b)
目标数据集: 4色 + 8类 分离格式

用法:
    python convert_dataset.py --src-images /path/to/source/images \
                               --src-labels /path/to/source/labels \
                               --dst-images /path/to/dest/images \
                               --dst-labels /path/to/dest/labels
"""
import argparse
from pathlib import Path
import shutil


# 源数据集52类名称（从other/rmarmor-pose.yaml）
SRC_NAMES = [
    "BGs", "B1b", "B2s", "B3s", "B4s", "B5s", "BOs", "BBs", "BBb", "BGb", "B3b", "B4b", "B5b",
    "RGs", "R1b", "R2s", "R3s", "R4s", "R5s", "ROs", "RBs", "RBb", "RGb", "R3b", "R4b", "R5b",
    "NGs", "N1s", "N2s", "N3s", "N4s", "N5s", "NO",  "NBs", "NBb", "NGb", "N3b", "N4b", "N5b",
    "PGs", "P1s", "P2s", "P3s", "P4s", "P5s", "POs", "PBs", "PBb", "PGb", "P3b", "P4b", "P5b",
]

# 目标类别
DST_COLORS = ["B", "R", "N", "P"]
DST_CLASSES = ["Gs", "1b", "2s", "3s", "4s", "O", "Bs", "Bb"]


def build_mapping():
    """构建源类别索引到目标(color, cls)的映射"""
    mapping = {}
    for src_idx, name in enumerate(SRC_NAMES):
        # 解析颜色
        color_char = name[0]  # B/R/N/P
        if color_char not in DST_COLORS:
            continue
        color_idx = DST_COLORS.index(color_char)

        # 解析类别+size
        cls_size = name[1:]  # 如 "Gs", "1b", "Os", "Bs", "Bb"

        # O类特殊处理：不区分size，统一映射到"O"
        if cls_size in ["Os", "O"]:
            cls_size = "O"

        # 检查是否在目标类别中
        if cls_size not in DST_CLASSES:
            continue

        cls_idx = DST_CLASSES.index(cls_size)
        mapping[src_idx] = (color_idx, cls_idx)

    return mapping


def convert_label_file(src_path, dst_path, mapping):
    """
    转换单个标签文件

    源格式: class_id x1 y1 x2 y2 x3 y3 x4 y4
    目标格式: color cls x1 y1 x2 y2 x3 y3 x4 y4

    返回: (成功转换的标注数, 跳过的标注数)
    """
    converted_lines = []
    skipped = 0

    with open(src_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                skipped += 1
                continue

            src_cls = int(parts[0])
            if src_cls not in mapping:
                skipped += 1
                continue  # 剔除不在目标范围内的标注

            color_idx, cls_idx = mapping[src_cls]
            coords = ' '.join(parts[1:])
            converted_lines.append(f"{color_idx} {cls_idx} {coords}")

    # 只有当有有效标注时才写入
    if converted_lines:
        with open(dst_path, 'w') as f:
            f.write('\n'.join(converted_lines) + '\n')
        return len(converted_lines), skipped
    return 0, skipped


def find_image_file(src_images, stem):
    """查找对应的图片文件（支持多种格式）"""
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
        candidate = src_images / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(
        description='将外部数据集格式转换为本项目格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python convert_dataset.py \\
        --src-images /data/external/images \\
        --src-labels /data/external/labels \\
        --dst-images ./converted/images \\
        --dst-labels ./converted/labels
        """
    )
    parser.add_argument('--src-images', required=True, help='源图片目录')
    parser.add_argument('--src-labels', required=True, help='源标签目录')
    parser.add_argument('--dst-images', required=True, help='目标图片目录')
    parser.add_argument('--dst-labels', required=True, help='目标标签目录')
    parser.add_argument('--dry-run', action='store_true', help='只统计不实际转换')
    args = parser.parse_args()

    src_images = Path(args.src_images)
    src_labels = Path(args.src_labels)
    dst_images = Path(args.dst_images)
    dst_labels = Path(args.dst_labels)

    # 检查源目录是否存在
    if not src_images.exists():
        print(f"错误: 源图片目录不存在: {src_images}")
        return 1
    if not src_labels.exists():
        print(f"错误: 源标签目录不存在: {src_labels}")
        return 1

    # 创建目标目录
    if not args.dry_run:
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

    mapping = build_mapping()
    print(f"类别映射表已构建，共 {len(mapping)} 个有效映射")

    converted_files = 0
    skipped_files = 0
    total_annotations = 0
    skipped_annotations = 0

    label_files = list(src_labels.glob('*.txt'))
    print(f"找到 {len(label_files)} 个标签文件")

    for label_file in label_files:
        image_stem = label_file.stem
        image_file = find_image_file(src_images, image_stem)

        if image_file is None:
            skipped_files += 1
            continue

        if args.dry_run:
            # 只统计
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 9:
                        src_cls = int(parts[0])
                        if src_cls in mapping:
                            total_annotations += 1
                        else:
                            skipped_annotations += 1
            converted_files += 1
        else:
            dst_label = dst_labels / label_file.name
            num_converted, num_skipped = convert_label_file(label_file, dst_label, mapping)

            if num_converted > 0:
                # 复制图片
                shutil.copy2(image_file, dst_images / image_file.name)
                converted_files += 1
                total_annotations += num_converted
                skipped_annotations += num_skipped
            else:
                skipped_files += 1
                skipped_annotations += num_skipped

    print()
    print("=" * 50)
    if args.dry_run:
        print("统计结果 (dry-run模式，未实际转换):")
    else:
        print("转换完成:")
    print(f"  转换文件: {converted_files}")
    print(f"  跳过文件: {skipped_files} (无有效标注或缺少图片)")
    print(f"  转换标注: {total_annotations}")
    print(f"  跳过标注: {skipped_annotations} (不在目标类别范围内)")
    print("=" * 50)

    return 0


if __name__ == '__main__':
    exit(main())
