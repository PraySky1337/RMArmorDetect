"""装甲板几何约束过滤器 - 基于实际数据分析"""

import torch
import numpy as np


class ArmorGeometryFilter:
    """
    装甲板几何约束过滤器

    基于667张训练图片的统计分析：
    - 宽高比中位数：2.12 (范围: 0.67-4.52, 95%在1.18-3.39)
    - 平行度：95%样本 < 1.8°
    - 100%为凸四边形
    - 相对面积：中位数0.12% (范围: 0.013%-3.5%)
    """

    def __init__(
        self,
        min_aspect_ratio: float = 0.80,   # 略宽松于数据下限(0.85)，排除单灯条(<1)
        max_aspect_ratio: float = 5.0,    # 略宽松于数据上限(4.19)
        min_relative_area: float = 0.0001,  # 0.01% - 过滤噪点
        max_relative_area: float = 0.05,    # 5% - 过滤异常大检测
        min_edge_length: float = 8.0,       # 像素 - 基于数据最小值8.7px
        max_parallel_angle_diff: float = 10.0,  # 度 - 宽松于数据(95%<1.8°)
        require_convex: bool = True,        # 必须为凸四边形
    ):
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_relative_area = min_relative_area
        self.max_relative_area = max_relative_area
        self.min_edge_length = min_edge_length
        self.max_parallel_angle_diff = max_parallel_angle_diff
        self.require_convex = require_convex

    def __call__(self, predictions, kpts, img_shape, return_mask=False):
        """
        过滤NMS后的预测结果

        Args:
            predictions: (N, 6+) - [x1, y1, x2, y2, conf, cls, ...]
            kpts: (N, 4, 2) or (N, 8) - 4个关键点坐标
            img_shape: (H, W) - 图像尺寸
            return_mask: 是否返回mask而不是过滤后的结果

        Returns:
            如果return_mask=False: (filtered_predictions, filtered_kpts)
            如果return_mask=True: keep_mask (N,) bool tensor
        """
        if predictions is None or len(predictions) == 0:
            if return_mask:
                return torch.ones(0, dtype=torch.bool, device=predictions.device)
            return predictions, kpts

        # 处理关键点格式
        if kpts.ndim == 2 and kpts.shape[1] == 8:
            kpts = kpts.reshape(-1, 4, 2)

        img_h, img_w = img_shape
        img_area = img_h * img_w

        keep_mask = torch.ones(len(predictions), dtype=torch.bool, device=predictions.device)

        for i in range(len(predictions)):
            kpt = kpts[i]  # (4, 2)

            # 1. 检查宽高比
            x_coords = kpt[:, 0]
            y_coords = kpt[:, 1]
            width = (x_coords.max() - x_coords.min()).item()
            height = (y_coords.max() - y_coords.min()).item()

            if height < 1e-3:  # 避免除零
                keep_mask[i] = False
                continue

            aspect_ratio = width / height
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                keep_mask[i] = False
                continue

            # 2. 检查面积
            area = self._compute_polygon_area(kpt)
            relative_area = area / img_area
            if relative_area < self.min_relative_area or relative_area > self.max_relative_area:
                keep_mask[i] = False
                continue

            # 3. 检查最小边长
            min_edge = self._compute_min_edge_length(kpt)
            if min_edge < self.min_edge_length:
                keep_mask[i] = False
                continue

            # 4. 检查凸性
            if self.require_convex and not self._is_convex(kpt):
                keep_mask[i] = False
                continue

            # 5. 检查平行度（装甲板特有）
            angle_diff = self._compute_parallel_angle_diff(kpt)
            if angle_diff > self.max_parallel_angle_diff:
                keep_mask[i] = False
                continue

        # 返回mask或过滤后的结果
        if return_mask:
            return keep_mask

        filtered_predictions = predictions[keep_mask]
        filtered_kpts = kpts[keep_mask]

        return filtered_predictions, filtered_kpts

    @staticmethod
    def _compute_polygon_area(kpt):
        """使用Shoelace公式计算四边形面积"""
        x = kpt[:, 0]
        y = kpt[:, 1]
        area = 0.5 * abs(
            x[0]*y[1] - x[1]*y[0] +
            x[1]*y[2] - x[2]*y[1] +
            x[2]*y[3] - x[3]*y[2] +
            x[3]*y[0] - x[0]*y[3]
        )
        return area.item()

    @staticmethod
    def _compute_min_edge_length(kpt):
        """计算最小边长"""
        edges = []
        for i in range(4):
            j = (i + 1) % 4
            edge_len = torch.norm(kpt[j] - kpt[i])
            edges.append(edge_len.item())
        return min(edges)

    @staticmethod
    def _is_convex(kpt):
        """检查是否为凸四边形（叉积法）"""
        def cross_product_sign(p1, p2, p3):
            v1 = p2 - p1
            v2 = p3 - p2
            return v1[0] * v2[1] - v1[1] * v2[0]

        signs = []
        for i in range(4):
            sign = cross_product_sign(kpt[i], kpt[(i+1)%4], kpt[(i+2)%4])
            signs.append(sign > 0)

        # 所有叉积同号 -> 凸四边形
        return all(signs) or not any(signs)

    @staticmethod
    def _compute_parallel_angle_diff(kpt):
        """
        计算平行度（上下边的角度差异）

        假设点顺序：0-左上，1-右上，2-右下，3-左下
        检查边0-1和边3-2的夹角
        """
        # 上边向量
        top_vec = kpt[1] - kpt[0]
        # 下边向量
        bottom_vec = kpt[2] - kpt[3]

        # 计算角度
        top_angle = torch.atan2(top_vec[1], top_vec[0]) * 180 / np.pi
        bottom_angle = torch.atan2(bottom_vec[1], bottom_vec[0]) * 180 / np.pi

        # 角度差
        angle_diff = abs(top_angle - bottom_angle).item()
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        return angle_diff


def apply_armor_filter(predictions, kpts, img_shape, **filter_kwargs):
    """
    便捷函数：应用装甲板几何过滤

    Args:
        predictions: (N, 6+) - NMS后的预测
        kpts: (N, 4, 2) or (N, 8) - 关键点
        img_shape: (H, W)
        **filter_kwargs: ArmorGeometryFilter的参数

    Returns:
        filtered_predictions, filtered_kpts
    """
    filter_fn = ArmorGeometryFilter(**filter_kwargs)
    return filter_fn(predictions, kpts, img_shape)
