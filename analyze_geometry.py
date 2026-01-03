#!/usr/bin/env python3
"""分析装甲板数据集的几何特征"""

import numpy as np
from pathlib import Path
import cv2

def parse_label_line(line):
    """解析标注行: color_id class_id x1 y1 x2 y2 x3 y3 x4 y4"""
    parts = line.strip().split()
    if len(parts) < 10:
        return None

    color_id = int(parts[0])
    class_id = int(parts[1])
    kpts = np.array([float(x) for x in parts[2:10]]).reshape(4, 2)
    return color_id, class_id, kpts

def compute_geometry(kpts, img_w, img_h):
    """计算几何特征（输入归一化坐标，输出像素坐标特征）"""
    # 转换为像素坐标
    kpts_px = kpts.copy()
    kpts_px[:, 0] *= img_w
    kpts_px[:, 1] *= img_h

    # 计算外接矩形
    x_min, y_min = kpts_px.min(axis=0)
    x_max, y_max = kpts_px.max(axis=0)
    width = x_max - x_min
    height = y_max - y_min

    # 宽高比
    aspect_ratio = width / (height + 1e-6)

    # 面积（使用Shoelace公式计算四边形面积）
    # 确保顺时针或逆时针顺序
    x = kpts_px[:, 0]
    y = kpts_px[:, 1]
    area = 0.5 * abs(
        x[0]*y[1] - x[1]*y[0] +
        x[1]*y[2] - x[2]*y[1] +
        x[2]*y[3] - x[3]*y[2] +
        x[3]*y[0] - x[0]*y[3]
    )

    # 相对面积（占图像的比例）
    img_area = img_w * img_h
    relative_area = area / img_area

    # 计算边长
    edges = []
    for i in range(4):
        j = (i + 1) % 4
        edge_len = np.linalg.norm(kpts_px[j] - kpts_px[i])
        edges.append(edge_len)

    min_edge = min(edges)
    max_edge = max(edges)

    # 检查是否为凸四边形（叉积法）
    def cross_product_sign(p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p2
        return v1[0] * v2[1] - v1[1] * v2[0]

    signs = []
    for i in range(4):
        sign = cross_product_sign(kpts_px[i], kpts_px[(i+1)%4], kpts_px[(i+2)%4])
        signs.append(sign > 0)

    is_convex = all(signs) or not any(signs)

    # 计算平行度（上下边的角度差异）
    # 假设点顺序：0-左上，1-右上，2-右下，3-左下（或类似）
    # 计算边0-1和边3-2的角度
    top_vec = kpts_px[1] - kpts_px[0]
    bottom_vec = kpts_px[2] - kpts_px[3]

    top_angle = np.arctan2(top_vec[1], top_vec[0]) * 180 / np.pi
    bottom_angle = np.arctan2(bottom_vec[1], bottom_vec[0]) * 180 / np.pi

    angle_diff = abs(top_angle - bottom_angle)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    return {
        'aspect_ratio': aspect_ratio,
        'width_px': width,
        'height_px': height,
        'area_px': area,
        'relative_area': relative_area,
        'min_edge': min_edge,
        'max_edge': max_edge,
        'is_convex': is_convex,
        'parallel_angle_diff': angle_diff,
    }

def analyze_dataset(dataset_root):
    """分析整个数据集"""
    dataset_root = Path(dataset_root)
    labels_dir = dataset_root / 'labels'
    images_dir = dataset_root / 'images'

    # 读取训练集列表
    train_list = dataset_root / 'splits' / 'train.txt'
    with open(train_list) as f:
        train_images = [line.strip() for line in f]

    all_stats = []

    print(f"分析 {len(train_images)} 张训练图片...")

    for idx, img_path_str in enumerate(train_images[:200]):  # 先分析前200张
        if idx % 20 == 0:
            print(f"  进度: {idx}/200...")
        img_path = Path(img_path_str)
        label_path = labels_dir / (img_path.stem + '.txt')

        if not label_path.exists():
            continue

        # 读取图片尺寸
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # 读取标注
        with open(label_path) as f:
            for line in f:
                parsed = parse_label_line(line)
                if parsed is None:
                    continue

                color_id, class_id, kpts = parsed
                geom = compute_geometry(kpts, img_w, img_h)
                geom['color_id'] = color_id
                geom['class_id'] = class_id
                geom['img_w'] = img_w
                geom['img_h'] = img_h
                all_stats.append(geom)

    return all_stats

def print_statistics(stats):
    """打印统计结果"""
    if not stats:
        print("没有数据！")
        return

    aspect_ratios = [s['aspect_ratio'] for s in stats]
    relative_areas = [s['relative_area'] for s in stats]
    min_edges = [s['min_edge'] for s in stats]
    parallel_diffs = [s['parallel_angle_diff'] for s in stats]
    convex_count = sum(1 for s in stats if s['is_convex'])

    print("\n" + "="*60)
    print("装甲板几何特征统计")
    print("="*60)
    print(f"总样本数: {len(stats)}")
    print(f"\n【宽高比 (Width/Height)】")
    print(f"  最小值: {np.min(aspect_ratios):.3f}")
    print(f"  最大值: {np.max(aspect_ratios):.3f}")
    print(f"  平均值: {np.mean(aspect_ratios):.3f}")
    print(f"  中位数: {np.median(aspect_ratios):.3f}")
    print(f"  5%分位: {np.percentile(aspect_ratios, 5):.3f}")
    print(f"  95%分位: {np.percentile(aspect_ratios, 95):.3f}")

    print(f"\n【相对面积 (占图像比例)】")
    print(f"  最小值: {np.min(relative_areas)*100:.4f}%")
    print(f"  最大值: {np.max(relative_areas)*100:.4f}%")
    print(f"  平均值: {np.mean(relative_areas)*100:.4f}%")
    print(f"  中位数: {np.median(relative_areas)*100:.4f}%")
    print(f"  5%分位: {np.percentile(relative_areas, 5)*100:.4f}%")
    print(f"  95%分位: {np.percentile(relative_areas, 95)*100:.4f}%")

    print(f"\n【最小边长 (像素)】")
    print(f"  最小值: {np.min(min_edges):.1f}px")
    print(f"  最大值: {np.max(min_edges):.1f}px")
    print(f"  平均值: {np.mean(min_edges):.1f}px")
    print(f"  5%分位: {np.percentile(min_edges, 5):.1f}px")

    print(f"\n【平行度 (上下边角度差)】")
    print(f"  最小值: {np.min(parallel_diffs):.1f}°")
    print(f"  最大值: {np.max(parallel_diffs):.1f}°")
    print(f"  平均值: {np.mean(parallel_diffs):.1f}°")
    print(f"  中位数: {np.median(parallel_diffs):.1f}°")
    print(f"  95%分位: {np.percentile(parallel_diffs, 95):.1f}°")

    print(f"\n【四边形形状】")
    print(f"  凸四边形: {convex_count}/{len(stats)} ({convex_count/len(stats)*100:.1f}%)")

    print("\n" + "="*60)
    print("建议的几何约束参数")
    print("="*60)

    # 基于统计给出建议
    ar_min = max(0.5, np.percentile(aspect_ratios, 1))  # 1%分位，至少0.5
    ar_max = min(10.0, np.percentile(aspect_ratios, 99))  # 99%分位，最多10.0

    area_min = max(0.0001, np.percentile(relative_areas, 1))  # 1%分位
    area_max = min(0.5, np.percentile(relative_areas, 99))  # 99%分位

    min_edge_threshold = max(5.0, np.percentile(min_edges, 1))  # 1%分位

    parallel_threshold = min(30.0, np.percentile(parallel_diffs, 95))  # 95%分位

    print(f"\nmin_aspect_ratio = {ar_min:.2f}  # 排除过窄的单灯条")
    print(f"max_aspect_ratio = {ar_max:.2f}  # 排除异常宽的检测")
    print(f"\nmin_relative_area = {area_min:.6f}  # 占图像{area_min*100:.4f}%")
    print(f"max_relative_area = {area_max:.6f}  # 占图像{area_max*100:.2f}%")
    print(f"\nmin_edge_length = {min_edge_threshold:.1f}  # 像素")
    print(f"\nmax_parallel_angle_diff = {parallel_threshold:.1f}  # 度")
    print(f"\nrequire_convex = True  # 必须为凸四边形")
    print("="*60 + "\n")

if __name__ == '__main__':
    dataset_root = '/home/ljy/Desktop/RMArmorDetect/datasets'
    stats = analyze_dataset(dataset_root)
    print_statistics(stats)
