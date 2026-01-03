"""Configuration files for armor detection."""

import os

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
ARMOR_YOLO11_CONFIG = os.path.join(CONFIG_DIR, 'armor_yolo11.yaml')
