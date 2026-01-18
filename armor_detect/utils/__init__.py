"""Utility functions for armor detection."""

# Constants
INITIAL_CONF = 0.1  # 提高初始置信度，改善训练后期conf偏低问题 (bias: -4.6 → -2.2)
TOP_K_ANCHORS = 9        # 减少正样本数量，集中学习信号
MAX_DIST_RATIO = 2.0     # 收紧距离阈值
FOCAL_GAMMA = 1.5

# Class names - 与数据集标签保持一致
# 数字类别 (nc_num=8): G, 1, 2, 3, 4, 5, O, B
CLASS_NAMES = ["G", "1", "2", "3", "4", "5", "O", "B"]
# 颜色类别 (nc_color=4): Blue, Red, Neutral, Purple
COLOR_NAMES = ["B", "R", "N", "P"]
# 尺寸类别 (nc_size=2): small, big
SIZE_NAMES = ["s", "b"]
