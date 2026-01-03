"""Utility functions for armor detection."""

# Constants
INITIAL_CONF = 0.01
TOP_K_ANCHORS = 5
MAX_DIST_RATIO = 1.2
FOCAL_GAMMA = 1.5

# Class names
NUMBER_CLASSES = ["Gs", "1b", "2s", "3s", "4s", "Os", "Bs", "Bb"]
COLOR_CLASSES = ["B", "R", "N", "P"]  # Blue, Red, Neutral, Purple
