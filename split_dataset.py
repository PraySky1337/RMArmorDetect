#!/usr/bin/env python3
"""
随机拆分数据集为训练集和验证集
用法: python split_dataset.py [datasets_dir] [--ratio VAL_RATIO]
默认: datasets_dir为同目录下的datasets文件夹, VAL_RATIO=0.1 (10%验证集)
"""

import argparse
import os
import shutil
from pathlib import Path
import random


def split_dataset(datasets_dir: str, val_ratio: float = 0.1, seed: int = 42):
    """
    将数据集随机拆分为train和val集

    Args:
        datasets_dir: 数据集目录路径
        val_ratio: 验证集比例 (0-1)
        seed: 随机种子
    """
    datasets_path = Path(datasets_dir)

    # 检查目录结构
    images_dir = datasets_path / "images"
    labels_dir = datasets_path / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"找不到images目录: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"找不到labels目录: {labels_dir}")

    # 获取所有图片文件
    image_files = list(images_dir.glob("*"))
    image_files = [f for f in image_files if f.is_file()]

    # 过滤出支持的图片格式
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in image_files if f.suffix.lower() in valid_extensions]

    if not image_files:
        raise ValueError("images目录中没有找到有效的图片文件")

    # 设置随机种子并打乱
    random.seed(seed)
    random.shuffle(image_files)

    # 计算分割点
    val_size = int(len(image_files) * val_ratio)
    val_files = image_files[:val_size]
    train_files = image_files[val_size:]

    print(f"总图片数: {len(image_files)}")
    print(f"训练集: {len(train_files)} ({(1-val_ratio)*100:.1f}%)")
    print(f"验证集: {len(val_files)} ({val_ratio*100:.1f}%)")

    # 创建目标目录
    for split in ["train", "val"]:
        (datasets_path / split / "images").mkdir(parents=True, exist_ok=True)
        (datasets_path / split / "labels").mkdir(parents=True, exist_ok=True)

    # 移动文件
    def move_files(files, split_name):
        for img_file in files:
            # 图片文件
            dst_img = datasets_path / split_name / "images" / img_file.name
            if img_file.exists():
                shutil.move(str(img_file), str(dst_img))

            # 对应的标签文件 (假设标签文件与图片同名但扩展名为.txt)
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                dst_label = datasets_path / split_name / "labels" / label_file.name
                shutil.move(str(label_file), str(dst_label))

    print("\n移动训练集文件...")
    move_files(train_files, "train")

    print("移动验证集文件...")
    move_files(val_files, "val")

    print("\n完成! 数据集已拆分到:")
    print(f"  - {datasets_path / 'train'}")
    print(f"  - {datasets_path / 'val'}")


def main():
    parser = argparse.ArgumentParser(description="随机拆分数据集为训练集和验证集")
    parser.add_argument(
        "datasets_dir",
        nargs="?",
        default=str(Path(__file__).parent / "datasets"),
        help="数据集目录路径 (默认: 同目录下的datasets文件夹)"
    )
    parser.add_argument(
        "--ratio", "-r",
        type=float,
        default=0.1,
        help="验证集比例, 0-1之间 (默认: 0.1, 即10%%)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )

    args = parser.parse_args()

    if not 0 < args.ratio < 1:
        parser.error("--ratio 必须在 0 和 1 之间")

    split_dataset(args.datasets_dir, args.ratio, args.seed)


if __name__ == "__main__":
    main()
